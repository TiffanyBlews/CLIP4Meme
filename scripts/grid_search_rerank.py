#!/usr/bin/env python3
import argparse
import itertools
import os
import subprocess
import sys
import time
import re
from datetime import datetime
from typing import Dict, Tuple, Any

def build_train_command(
    docker_name: str,
    conda_env: str,
    workdir_in_container: str,
    gpu_id: int,
    base_args: dict,
    lr: float,
    mlp_hidden_dim: int,
    mlp_layers: int,
    mlp_dropout: float,
    extra_negatives: int,
    log_dir_host: str,
    python_path: str = "",
):
    suffix = f"lr{lr}_hd{mlp_hidden_dim}_ly{mlp_layers}_do{mlp_dropout}_neg{extra_negatives}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"grid_{timestamp}_{suffix}.log"
    log_path_container = os.path.join(workdir_in_container, "logs", log_name)
    log_path_host = os.path.join(log_dir_host, log_name)
    os.makedirs(log_dir_host, exist_ok=True)

    # Ensure 'logs' exists inside container before redirect, create lazily using bash
    inner_cmd = (
        f"mkdir -p {os.path.join(workdir_in_container, 'logs')} && "
        f"cd {workdir_in_container} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python models/rerank/rerank_train.py "
        f"--lr {lr} "
        f"--epochs {base_args['epochs']} "
        f"--batch_size {base_args['batch_size']} "
        f"--model_type mlp "
        f"--loss_type lambdarank "
        f"--label_mode {base_args['label_mode']} "
        f"--clip_checkpoint {base_args['clip_checkpoint']} "
        f"--topk_base {base_args['topk_base']} "
        f"--extra_negatives {extra_negatives} "
        f"--save_dir {base_args['save_dir']} "
        f"--wandb_project {base_args['wandb_project']} "
        f"--wandb_entity {base_args.get('wandb_entity','')} "
        f"--wandb_mode {base_args.get('wandb_mode','disabled')} "
        f"--keep_all_best "
        f"--mlp_hidden_dim {mlp_hidden_dim} "
        f"--mlp_layers {mlp_layers} "
        f"--mlp_dropout {mlp_dropout} "
        f"2>&1 | tee {log_path_container}"
    )

    # conda activation needs login shell semantics
    if python_path:
        docker_exec_cmd = [
            "docker", "exec", "-i", docker_name,
            "bash", "-lc",
            f"mkdir -p {os.path.join(workdir_in_container, 'logs')} && cd {workdir_in_container} && CUDA_VISIBLE_DEVICES={gpu_id} {python_path} models/rerank/rerank_train.py --lr {lr} --epochs {base_args['epochs']} --batch_size {base_args['batch_size']} --model_type mlp --loss_type lambdarank --label_mode {base_args['label_mode']} --clip_checkpoint {base_args['clip_checkpoint']} --topk_base {base_args['topk_base']} --extra_negatives {extra_negatives} --save_dir {base_args['save_dir']} --wandb_project {base_args['wandb_project']} --wandb_entity {base_args.get('wandb_entity','')} --wandb_mode {base_args.get('wandb_mode','disabled')} --keep_all_best --mlp_hidden_dim {mlp_hidden_dim} --mlp_layers {mlp_layers} --mlp_dropout {mlp_dropout} 2>&1 | tee {log_path_container}"
        ]
    else:
        docker_exec_cmd = [
            "docker", "exec", "-i", docker_name,
            "bash", "-lc",
            f"(eval \"$(conda shell.bash hook)\" || source ~/.bashrc) && conda activate {conda_env} && {inner_cmd}"
        ]
    return docker_exec_cmd, log_path_host

def parse_metrics_from_log(log_text: str):
    # Try to capture the last printed metrics dict or lines
    # Pattern for dict print: {'R@1': 12.3, 'P@1': ...}
    dict_pattern = re.compile(r"\{[^}]*R@1[^}]*\}")
    matches = dict_pattern.findall(log_text)
    metrics = {}
    if matches:
        # Try eval safely by replacing nan/inf if present
        s = matches[-1]
        try:
            # Convert to Python dict by removing potential numpy formatting
            metrics = eval(s, {"__builtins__": {}}, {})
        except Exception:
            metrics = {}
    else:
        # Fallback: look for lines like "R@1: 12.34% | P@1: 8.90%"
        line_pattern = re.compile(r"R@(\d+):\s*([0-9.]+)")
        for k, v in line_pattern.findall(log_text):
            metrics[f"R@{k}"] = float(v)
    return metrics

def run_combo(
    args,
    base_args,
    lr_val,
    hd,
    ly,
    do,
    neg,
    cache: Dict[Tuple[Any, ...], Dict[str, Any]],
):
    key = (lr_val, hd, ly, do, neg)
    if key in cache:
        return cache[key]
    docker_cmd, log_host_path = build_train_command(
        docker_name=args.docker_name,
        conda_env=args.conda_env,
        workdir_in_container=args.workdir_in_container,
        gpu_id=args.gpu_id,
        base_args=base_args,
        lr=lr_val,
        mlp_hidden_dim=hd,
        mlp_layers=ly,
        mlp_dropout=do,
        extra_negatives=neg,
        log_dir_host=args.log_dir_host,
        python_path=args.python_path,
    )
    print("CMD:", " ".join(docker_cmd))
    if args.dry_run:
        res = {"R@1": None, "metrics": {}, "log": log_host_path, "hidden_dim": hd, "layers": ly, "dropout": do, "extra_negatives": neg, "lr": lr_val}
        cache[key] = res
        return res
    os.makedirs(args.log_dir_host, exist_ok=True)
    with open(log_host_path, "wb") as fout:
        proc = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            chunk = proc.stdout.readline()
            if not chunk:
                break
            fout.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        ret = proc.wait()
        print(f"Return code: {ret}")
    try:
        with open(log_host_path, "r", encoding="utf-8", errors="ignore") as fin:
            txt = fin.read()
        metrics = parse_metrics_from_log(txt)
    except Exception:
        metrics = {}
    r1 = metrics.get("R@1") or metrics.get("base_R@1") or metrics.get("rerank_R@1")
    res = {
        "hidden_dim": hd,
        "layers": ly,
        "dropout": do,
        "extra_negatives": neg,
        "lr": lr_val,
        "metrics": metrics,
        "R@1": r1,
        "log": log_host_path,
    }
    cache[key] = res
    time.sleep(2)
    return res

def main():
    parser = argparse.ArgumentParser(description="Grid search for rerank MLP in Docker")
    parser.add_argument("--docker_name", type=str, default="cuda_1111")
    parser.add_argument("--conda_env", type=str, default="neo_meme")
    parser.add_argument("--workdir_in_container", type=str, default="/root/ljj", help="Path of project inside container")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--label_mode", type=str, default="inv_rank")
    parser.add_argument("--clip_checkpoint", type=str, default="./checkpoints_clip/clip_imgflip.pt")
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rerank")
    parser.add_argument("--wandb_project", type=str, default="meme_rerank")
    parser.add_argument("--wandb_entity", type=str, default="wangtn")
    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument("--grid_hidden_dim", type=int, nargs="+", default=[512, 768])
    parser.add_argument("--grid_layers", type=int, nargs="+", default=[3, 5,  7])
    parser.add_argument("--grid_dropout", type=float, nargs="+", default=[0.5])
    parser.add_argument("--grid_extra_negatives", type=int, nargs="+", default=[50])
    parser.add_argument("--grid_lr", type=float, nargs="+", default=[1e-5])
    parser.add_argument("--search_mode", type=str, choices=["grid", "greedy"], default="grid")
    parser.add_argument("--stage_order", type=str, default="lr,hidden_dim,layers,extra_negatives")

    parser.add_argument("--log_dir_host", type=str, default="./grid_logs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python_path", type=str, default="/home/nlper/anaconda3/envs/neo_meme/bin/python")
    args = parser.parse_args()

    base_args = {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "label_mode": args.label_mode,
        "clip_checkpoint": args.clip_checkpoint,
        "topk_base": args.topk_base,
        "save_dir": args.save_dir,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
    }

    cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    results = []
    if args.search_mode == "grid":
        total = 0
        for lr_val, hd, ly, do, neg in itertools.product(
            args.grid_lr,
            args.grid_hidden_dim,
            args.grid_layers,
            args.grid_dropout,
            args.grid_extra_negatives,
        ):
            total += 1
            print(f"[{total}] lr={lr_val} hd={hd} layers={ly} dropout={do} extra_neg={neg}")
            res = run_combo(args, base_args, lr_val, hd, ly, do, neg, cache)
            results.append(res)
        results_sorted = sorted(results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
        print("\n=== Grid Search Summary (sorted by R@1) ===")
        for i, r in enumerate(results_sorted, 1):
            print(f"{i}. lr={r['lr']} hd={r['hidden_dim']} ly={r['layers']} do={r['dropout']} neg={r['extra_negatives']} R@1={r['R@1']} log={r['log']}")
        return

    order = [s.strip() for s in args.stage_order.split(",") if s.strip()]
    current_lr = args.lr if args.lr else (args.grid_lr[0] if args.grid_lr else 1e-5)
    current_hd = args.grid_hidden_dim[0]
    current_ly = args.grid_layers[0]
    current_do = args.grid_dropout[0]
    current_neg = args.grid_extra_negatives[0]
    best_summary = []

    for stage in order:
        if stage == "hidden_dim":
            candidates = args.grid_hidden_dim
            stage_results = []
            for hd in candidates:
                r = run_combo(args, base_args, current_lr, hd, current_ly, current_do, current_neg, cache)
                stage_results.append(r)
            best = max(stage_results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
            current_hd = best["hidden_dim"]
            best_summary.append(("hidden_dim", current_hd, best["R@1"]))
        elif stage == "layers":
            candidates = args.grid_layers
            stage_results = []
            for ly in candidates:
                r = run_combo(args, base_args, current_lr, current_hd, ly, current_do, current_neg, cache)
                stage_results.append(r)
            best = max(stage_results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
            current_ly = best["layers"]
            best_summary.append(("layers", current_ly, best["R@1"]))
        elif stage == "dropout":
            candidates = args.grid_dropout
            stage_results = []
            for do in candidates:
                r = run_combo(args, base_args, current_lr, current_hd, current_ly, do, current_neg, cache)
                stage_results.append(r)
            best = max(stage_results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
            current_do = best["dropout"]
            best_summary.append(("dropout", current_do, best["R@1"]))
        elif stage == "extra_negatives":
            candidates = args.grid_extra_negatives
            stage_results = []
            for neg in candidates:
                r = run_combo(args, base_args, current_lr, current_hd, current_ly, current_do, neg, cache)
                stage_results.append(r)
            best = max(stage_results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
            current_neg = best["extra_negatives"]
            best_summary.append(("extra_negatives", current_neg, best["R@1"]))
        elif stage == "lr":
            candidates = args.grid_lr
            stage_results = []
            for lr_val in candidates:
                r = run_combo(args, base_args, lr_val, current_hd, current_ly, current_do, current_neg, cache)
                stage_results.append(r)
            best = max(stage_results, key=lambda x: (x["R@1"] is None, -(x["R@1"] or -1.0)))
            current_lr = best["lr"]
            best_summary.append(("lr", current_lr, best["R@1"]))

    final_res = run_combo(args, base_args, current_lr, current_hd, current_ly, current_do, current_neg, cache)
    print("\n=== Greedy Search Summary ===")
    for name, val, r1 in best_summary:
        print(f"{name}: {val} | R@1={r1}")
    print(f"final: lr={current_lr} hd={current_hd} ly={current_ly} do={current_do} neg={current_neg} | R@1={final_res['R@1']} log={final_res['log']}")

if __name__ == "__main__":
    main()
