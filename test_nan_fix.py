#!/usr/bin/env python3
"""
测试脚本：验证NaN问题是否已修复
"""

import torch
import torch.nn.functional as F
from model import EFB

def test_model_forward():
    """测试模型前向传播是否产生NaN"""
    print("测试模型前向传播...")
    
    # 创建模型
    model = EFB()
    model.eval()
    
    # 创建模拟输入数据
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟图像输入 (CLIP ViT-B/32 期望 224x224 图像)
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 模拟文本输入 (CLIP tokenizer 期望的格式)
    query = torch.randint(0, 49408, (batch_size, 77)).to(device)  # CLIP vocab size
    context = torch.randint(0, 49408, (batch_size, 77)).to(device)
    
    # 模拟情感输入 (BERT tokenizer 期望的格式)
    emotion_ids = torch.randint(0, 21128, (batch_size, 32)).to(device)  # BERT vocab size
    emotion_mask = torch.ones(batch_size, 32).to(device)
    
    # 移动模型到设备
    model = model.to(device)
    
    try:
        with torch.no_grad():
            # 前向传播
            outputs = model(image, query, context, emotion_ids, emotion_mask)
            
            sim_qi = outputs['sim_qi']
            sim_qc = outputs['sim_qc']
            
            # 检查是否有NaN
            if torch.isnan(sim_qi).any():
                print("❌ sim_qi 包含 NaN 值")
                return False
            else:
                print("✅ sim_qi 没有 NaN 值")
                
            if torch.isnan(sim_qc).any():
                print("❌ sim_qc 包含 NaN 值")
                return False
            else:
                print("✅ sim_qc 没有 NaN 值")
            
            # 打印相似度矩阵的范围
            print(f"sim_qi 范围: [{sim_qi.min().item():.4f}, {sim_qi.max().item():.4f}]")
            print(f"sim_qc 范围: [{sim_qc.min().item():.4f}, {sim_qc.max().item():.4f}]")
            
            # 测试损失计算
            def test_loss(sim_matrix):
                batch_size = sim_matrix.size(0)
                labels = torch.arange(batch_size, device=sim_matrix.device)
                loss_rows = F.cross_entropy(sim_matrix, labels)
                loss_cols = F.cross_entropy(sim_matrix.t(), labels)
                return (loss_rows + loss_cols) / 2.0
            
            loss_qi = test_loss(sim_qi)
            loss_qc = test_loss(sim_qc)
            total_loss = loss_qi + loss_qc
            
            if torch.isnan(total_loss):
                print("❌ 损失计算产生 NaN")
                return False
            else:
                print("✅ 损失计算正常")
                print(f"总损失: {total_loss.item():.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_logit_scale():
    """测试 logit_scale 参数"""
    print("\n测试 logit_scale 参数...")
    
    model = EFB()
    logit_scale = model.logit_scale
    
    print(f"logit_scale 初始值: {logit_scale.item():.4f}")
    print(f"exp(logit_scale): {logit_scale.exp().item():.4f}")
    
    # 检查是否在合理范围内
    if logit_scale.exp().item() > 50:
        print("⚠️  logit_scale 可能仍然过大")
    else:
        print("✅ logit_scale 在合理范围内")

if __name__ == "__main__":
    print("开始测试 NaN 修复...")
    
    test_logit_scale()
    success = test_model_forward()
    
    if success:
        print("\n🎉 所有测试通过！NaN 问题已修复。")
    else:
        print("\n❌ 测试失败，需要进一步调试。") 