# Generate data
python t2t-datagen.py \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --num_shards=100 \
  --problem=$PROBLEM

mv $TMP_DIR/tokens.vocab.32768 $DATA_DIR