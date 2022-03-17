# Bert_compression

## Prerequisite 

#### 1. Install packages
use the following command to install packages:
```sh
pip install pytorch_pretrained_bert
pip install transformers
```

The torch version is 1.10.0+cu111

#### 2. Download Processed GLUE Dataset 
I already preprocessed the GLUE Dataset with task `CoLA, RTE, STS-B, MRPC, SST-2, WNLI`, please download the dataset from: https://drive.google.com/drive/folders/1-DRrA5MVKI-RZlEwrbIAeencVcf6hjjs?usp=sharing, and place the unziped file in path `/Bert_compression/GLUE_data`.

## Experiment
The code is heavily based on https://github.com/maknotavailable/pytorch-pretrained-BERT
#### 1. Head pruning experiment
For classification tasks (CoLA, SST-2, MRPC), please run the following command in directory `Bert_compression/pytorch-pretrained-BERT/`:
```sh
##cola, mnli,mrpc,sst-2
python examples/run_classifier.py \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/GLUE_data/MRPC/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir  /path/to/models/MRPC/full 2>&1
```

For regression tasks (RTE, STS-B, WNLI), please run the following command in directory `Bert_compression/pytorch-pretrained-BERT/`:
```sh
python examples/run_regression.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name wnli \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /path/to/GLUE_data/WNLI \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir/path/to/models/WNLI \
  --overwrite_output_dir \
  
  ```
