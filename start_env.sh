docker exec -it cuda_1111 bash
conda activate neo_meme
cd ljj
nohup python train.py > 0809.log 2>&1 &