DATA_DIR=$1

python main.py \
  --model_name_or_path roberta-base \
  --task_name Rec \
  --do_train \
  --do_eval \
  --do_predict \
  --item_reviews_file $DATA_DIR/item_review.tsv \
  --user_reviews_file $DATA_DIR/user_review.tsv \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./saved_checkpoints_"$1"
