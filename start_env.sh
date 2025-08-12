docker exec -it cuda_1111 bash
conda activate neo_meme
cd ljj
nohup python train.py --stage 1 > 0811_stage1_imgflip.log 2>&1 &