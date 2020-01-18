export BERT_BASE_DIR=/home/lei/PycharmProjects/BERT_NRE/cased_L-12_H-768_A-12
export GLUE_DIR=/home/lei/PycharmProjects/BERT_NRE

python3 run_classifier.py \
  --do_train=True \
  --do_eval=False \
  --data_dir=$GLUE_DIR/data/nyt10/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --use_pcnn=False \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/reuslt/