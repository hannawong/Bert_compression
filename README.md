# Deployment of BERT in Edge Devices: Compression by Heads & Layer Pruning

###### by Zihan Wang (UNI: zw2782), Jace Yang (UNI: jy3174)



## 1. Introduction of this project

In this project, we firstly experiment different heads & layers pruning strategies upon BERT, and then deploy compressed BERT model to Google PIXEL 2 (4GB RAM) with acceptable inference time. In this part, we will give a brief introduction on the intuition of performing layer & head pruning in transformer models as well as the deployment on edge device.

### 1.1 Pruning layers in transformer models

For pretrained language models such as BERT, XLNet, Roberta, etc., there is an embedding layer and L encoder layers ![img](https://www.google.com/chart?cht=tx&chf=bg,s,FFFFFF00&chco=000000&chl=%5C%7B%7Bl%7D_%7B1%7D%2C%7Bl%7D_%7B2%7D%2C...%2C%7Bl%7D_%7BL%7D%5C%7D). However, according to the paper [Linguistic Knowledge and Transferability of Contextual Representations](https://www.google.com/url?q=https://www.google.com/url?q%3Dhttps://aclanthology.org/N19-1112.pdf%26amp;sa%3DD%26amp;source%3Deditors%26amp;ust%3D1650298256701385%26amp;usg%3DAOvVaw1-75S3t8d1ghJ7bS7tNn5E&sa=D&source=docs&ust=1650298256721863&usg=AOvVaw1lAnSdRO7B4_-HYAkT9BSz), different layers in transformer-based models capture different linguistic information. For example, lower layers of the network capture syntax information whereas higher-level information is learned at middle and higher layers in the network. These findings lead us to investigate the effect of dropping each layer during fine-tuning and testing on the test result. More specifically, we try different layer-dropping strategies, such as top-layer dropping, bottom-layer dropping, middle-layer dropping, and alternate dropping (odd alternate, even alternate).

### 1.2 Pruning attention heads in transformer models

In paper [Are Sixteen Heads Really Better than One?](https://www.google.com/url?q=https://www.google.com/url?q%3Dhttps://arxiv.org/pdf/1905.10650.pdf%26amp;sa%3DD%26amp;source%3Deditors%26amp;ust%3D1650298256701957%26amp;usg%3DAOvVaw1FFTIWpoKZSUzYtuIPlmuZ&sa=D&source=docs&ust=1650298256722026&usg=AOvVaw2tBkDzpZTxbp-pE4IDcfr2) researchers came to a surprising conclusion: most attention heads can be individually removed after training without any significant downside in terms of test performance. However, while attempting to answer the question "Are important heads 'universally' important across datasets?", they only employ two out-of-domain test sets, that are, from our perspective, not totally unrelated. Therefore, we believe that the conclusion that key heads are universally important across the dataset needs further experiment. Specifically, we perform ablation study by masking attention heads (replace attention output with zero-vector) in each layer during the testing stage using GLUE dataset, and then conduct a cross-task analysis to see how pruning affects testing accuracy.

### 1.3 Deployment on End-Device

 [Huggingface](https://huggingface.co/)'s [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) is a smaller and faster version of BERT. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. In this project, we quantize and deploy DistilBERT on Google Pixel 2 (Android 11.0 x86, 4GB RAM, 64GB Storage) to perform Question Answering, Sentiment Analysis(SST-2), Semantic Textual Similarity Analysis (STS-B) tasks within the space the inference time constraints.



For layer and heads pruning experiments, we perform task-specific fine-tuning using the [GLUE](https://www.google.com/url?q=https://www.google.com/url?q%3Dhttps://gluebenchmark.com/%26amp;sa%3DD%26amp;source%3Deditors%26amp;ust%3D1650298256704042%26amp;usg%3DAOvVaw18o0nZ27v6f6RnT7cQgrtJ&sa=D&source=docs&ust=1650298256722399&usg=AOvVaw0kMoeBsOiqVRx6trV6BavG) training set and evaluate testing metrics on the official dev set. Among a variety of language understanding tasks in GLUE, we report model performance on 6 tasks, i.e. **SST-2**(Sentiment Analysis), **CoLA**(Corpus of Linguistic Acceptability); **WNLI**(Winograd NLI), **STS-B**(Semantic Textual Similarity Benchmark), **RTE**(Recognizing Textual Entailment) and **MRPC**(Microsoft Research Paraphrase Corpus).



## 2. Intro of Repository Architecture

The Repository has three main directories: `pytorch-pretrained-BERT`, `AndroidApp`, and `logs`. `logs` is the directory that records our experiment result from which we draw our conclusion. In this section, we will introduce the other two directories in detail.

### 2.1 **pytorch-pretrained-BERT**

This repository contains an PyTorch reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. Our code is based on https://github.com/maknotavailable/pytorch-pretrained-BERT.

In order to perform heads pruning experiments, we rewrite the BERT implementation in  `pytorch-pretrained-BERT/pytorch_pretrained_bert/modeling.py`by adding functions related to heads_pruning (such as prune_heads, mask_heads) in `BertAttention` module.  We typically finetune the full Bert model on GLUE dataset, then prune each head (there are 12*12 = 144 heads in total) and test on GLUE test sets.   

In order to perform layer pruning experiments, we add argument `remove_layers` that takes argument of layer index and delete the parameters for corresponding layers in Bert. We use different layer dropping strategies such as top-layer dropping, bottom-layer dropping, middle-layer dropping, and alternate dropping. The pruned Bert is then used for finetuning and inferencing. 

In order to finetune the model and test it on GLUE benchmark, we also add the code for GLUE dataset preprocessing, as well as code for running classification tasks of GLUE in `pytorch-pretrained-BERT/examples/run_classifier.py` , and code for running regression tasks of GLUE in `pytorch-pretrained-BERT/examples/run_regression.py`.

### 2.2 AndroidApp

This directory is the deployment of compressed Bert model on Android edge device. The code is written in Kotlin.

We firstly quantize and convert the Huggingface's DistilBert QA model to TorchScript and deploy it on Android perform question answering. Then we use DistilBert finetuned on GLUE dataset and deploy it on Android to perform tasks such as Sentiment Analysis(SST-2) and Semantic Textual Similarity Analysis (STS-B).



## 3. How to Run Our Code

### 3.1 Prerequisite 

#### 3.1.2 Install packages
use the following command to install packages:
```sh
pip install pytorch_pretrained_bert
pip install transformers
```

The torch version is 1.10.0+cu111, and python version is 3.8. 

#### 3.2.2 Download Preprocessed GLUE Dataset 
I already preprocessed the GLUE Dataset with task `CoLA, RTE, STS-B, MRPC, SST-2, WNLI`, please download the dataset from: https://drive.google.com/drive/folders/1--7Tp6I7mWJsnONgc4Gv7ZYhRNy4fbse?usp=sharing, and place the unziped file in path `/Bert_compression/GLUE_data`.

### 3.2 Run Experiment
#### 3.2.1 Head pruning experiment
For classification tasks (CoLA, SST-2, MRPC), please run the following command in directory `Bert_compression/pytorch-pretrained-BERT/`:
```sh
##cola,mrpc,sst-2
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

In this way, we perform ablation study on each of the 12*12 attention heads iteratively, and show the test results on GLUE benchmark after removing each head. 

#### 3.2.2 Layer Pruning Experiment

classification tasks:

```
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
    --remove_layers "0,1,2,3" \
    --output_dir  /path/to/models/MRPC/full 2>&1
```



regression tasks:

```shell
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
  --remove_layers "0,1,2,3" \
  --overwrite_output_dir \
```



#### 3.2.3 Deployment on Edge Device

**(1) Question Answering Task**

Firstly, run `python convert_distilbert_qa.py` to generate quantized DistilBert model for QuestionAnswering.  After the script completes, copy the model file `qa360_quantized.ptl` to the Android app's assets folder.

![](https://github.com/hannawong/Bert_compression/blob/main/AndriodApp/QA.png)

**(2) Semantic Textual Similarity Analysis**

run `python convert_distilbert_sts.py` to generate quantized DistilBert model for Semantic Textual Similarity Analysis. 

![](https://github.com/hannawong/Bert_compression/blob/main/AndriodApp/similarity.gif)

**(3) Sentiment Classification Task**

Similarity, run `python convert_distilbert_sst.py` to generate quantized DistilBert model for sentiment analysis. 

![](https://github.com/hannawong/Bert_compression/blob/main/AndriodApp/sentiment.gif)
