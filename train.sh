DATA_PATH=/root/zt/data_topics
python efb_retrieval.py \
  --csv_path ${DATA_PATH}/input_file/train_video_id_9k.csv \
  --json_path  ${DATA_PATH}/input_file/train_9k.json \
  --features_path ${DATA_PATH}/image \
  --train_emotion ${DATA_PATH}/input_file/train_emotion.json \
  --test_json ${DATA_PATH}/input_file/test_4k.json  \
  --test_emotion ${DATA_PATH}/input_file/test_emotion.json \
  --batch_size 64 --batch_size_val 128 \
  --lr 1e-4 --epochs 6 --alpha 0.5 \
  --patience 2 --ckpt_dir ./checkpoints
