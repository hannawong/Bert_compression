##cola, mnli,mrpc,sst-2
!python examples/run_classifier.py \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /content/drive/MyDrive/Bert_compression/GLUE_data/MRPC/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /content/drive/MyDrive/Bert_compression/models/MRPC/full 2>&1