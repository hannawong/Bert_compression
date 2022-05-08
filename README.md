# Deploying BERT in Edge Devices: Compression by Heads Pruning & Layer Drop

> Final Project of Columbia COMS 6998 Practical Deep Learning System Performance, 2022 Spring. <br> By Zihan Wang (UNI: zw2782) and Jace Yang (UNI: jy3174)

## 1. Project Introduction

In this project, we experiment many different heads & layers pruning strategies upon BERT to obtrain a good latency-accuracy trade-off in depolying BERT to edge devices.

In this part, we will give a brief introduction on the intuition and background work behind pruning.

### 1.1 Pruning layers in transformer models

For pretrained language models such as BERT, XLNet, Roberta, etc., there is an embedding layer and L encoder layers <img src="https://render.githubusercontent.com/render/math?math=\{l_{1}, l_{2}, \dots, l, l_{L}\}">. However, according to the paper [Linguistic Knowledge and Transferability of Contextual Representations](https://arxiv.org/abs/1903.08855), different layers in transformer-based models capture different linguistic information. For example, lower layers of the network capture syntax information whereas higher-level information is learned at middle and higher layers in the network. 

These findings lead us to investigate the impact of dropping each layer during fine-tuning to the testing outcome. More specifically, we tried different layer-dropping strategies, such as top-layer dropping, bottom-layer dropping, and symmetric dropping.

### 1.2 Pruning attention heads in transformer models

In paper [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) researchers came to a surprising conclusion: most attention heads can be individually removed after training without any significant downside in terms of test performance. 

However, while attempting to answer the question "Are important heads 'universally' important across datasets?" in the chapter 3.4, researchers only employed two out-of-domain test sets, that are, from our perspective, not totally unrelated! 

Therefore, we believe that the conclusion that key heads are universally important across the dataset needs further experiment. Specifically, we perform ablation study by masking attention heads (replace attention output with zero-vector) in each layer during the testing stage using GLUE dataset, and then conduct a comprehensive analysis across different tasks to see how pruning affects testing accuracy.


For layer and heads pruning experiments, we perform task-specific fine-tuning using the [GLUE](https://gluebenchmark.com/leaderboard) training set and evaluate testing metrics on the official dev set. Among a variety of language understanding tasks in GLUE, we report model performance on 6 tasks, i.e. **SST-2**(Sentiment Analysis), **CoLA**(Corpus of Linguistic Acceptability); **WNLI**(Winograd NLI), **STS-B**(Semantic Textual Similarity Benchmark), **RTE**(Recognizing Textual Entailment) and **MRPC**(Microsoft Research Paraphrase Corpus).

### 1.3 Deployment on End-Device

 [Huggingface](https://huggingface.co/)'s [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) is a smaller and faster version of BERT. It has the same structure as our 6-layers dropped BERT so that could be used as a method to verify our idea. In this project, we quantized and deployed DistilBERT on Google Pixel 2 (Android 11.0 x86, 4GB RAM, 64GB Storage) to perform Question Answering, Sentiment Analysis(SST-2), Semantic Textual Similarity Analysis (STS-B) tasks within both the space and the inference time constraints.



## 2. Repository Architecture Description

The Repository has three main directories: `pytorch-pretrained-BERT`, `AndroidApp`, and `logs`. 


### 2.1 **pytorch-pretrained-BERT**

This repository contains an PyTorch reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. Our code is based on https://github.com/maknotavailable/pytorch-pretrained-BERT.

In order to perform heads pruning experiments, we rewrite the BERT implementation in  `pytorch-pretrained-BERT/pytorch_pretrained_bert/modeling.py`by adding functions related to heads_pruning (such as prune_heads, mask_heads) in `BertAttention` module.  We typically finetune the full Bert model on GLUE dataset, then prune each head (there are 12*12 = 144 heads in total) and test on GLUE test sets.   

In order to perform layer pruning experiments, we add argument `remove_layers` that takes argument of layer index and delete the parameters for corresponding layers in Bert. We use different layer dropping strategies such as top-layer dropping, bottom-layer dropping, middle-layer dropping, and alternate dropping. The pruned Bert is then used for finetuning and inferencing. 

In order to finetune the model and test it on GLUE benchmark, we also add the code for GLUE dataset preprocessing, as well as code for running classification tasks of GLUE in `pytorch-pretrained-BERT/examples/run_classifier.py` , and code for running regression tasks of GLUE in `pytorch-pretrained-BERT/examples/run_regression.py`.

### 2.2 AndroidApp

This directory is the deployment of compressed Bert model on Android edge device. The code is written in Kotlin.

We firstly quantize and convert the Huggingface's DistilBert QA model to TorchScript and deploy it on Android perform question answering. Then we use DistilBert finetuned on GLUE dataset and deploy it on Android to perform tasks such as Sentiment Analysis(SST-2) and Semantic Textual Similarity Analysis (STS-B).

### 2.3 Result analysis

`logs` is the output directory of *2.1*, which records our experiment results as well as codes to transform the log text, format data, and plot the graphs that our analysis rely on in *Part 4*.

## 3. Commands to Execute Our Code

### 3.1 Prerequisite 

#### 3.1.2 Install packages
use the following command to install packages:
```sh
pip install pytorch_pretrained_bert
pip install transformers
```

The torch version is 1.10.0+cu111, and python version is 3.8. 

#### 3.2.2 Download Preprocessed GLUE Dataset 
We already preprocessed the GLUE Dataset with task `CoLA, RTE, STS-B, MRPC, SST-2, WNLI`, please download the dataset from: https://drive.google.com/drive/folders/1--7Tp6I7mWJsnONgc4Gv7ZYhRNy4fbse?usp=sharing, and place the unziped file in directory path `Bert_compression/GLUE_data`.

### 3.2 Run Experiment

For attention heads pruning, since there are 12 layers and 12 heads for each layer, we need to perform 144 pruning experiments for testing stage for each of the 6 tasks. For layer drop, we experiment pruning 2, 4, or 6 layers with 3 different strategies, so that will be 9 experiments for each of the 6 tasks.


#### 3.2.1 Head pruning experiment
- For classification tasks (CoLA, SST-2, MRPC), please run the following command in directory `Bert_compression/pytorch-pretrained-BERT/`:

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

- For regression tasks (RTE, STS-B, WNLI), please run the following command in directory `Bert_compression/pytorch-pretrained-BERT/`:
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

This way, we performed ablation study on each of the 12 × 12 attention heads iteratively, and show the test results on GLUE benchmark after removing each head. 

#### 3.2.2 Layer Pruning Experiment

- Classification tasks:

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


- Regression tasks:

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

<center><img src="AndriodApp/QA.png" width="75%"/></center>

**(2) Semantic Textual Similarity Analysis**

Run `python convert_distilbert_sts.py` to generate quantized DistilBert model for Semantic Textual Similarity Analysis. 

<center><img src="AndriodApp/similarity.gif" width="75%"/></center>

**(3) Sentiment Classification Task**

Similarity, run `python convert_distilbert_sst.py` to generate quantized DistilBert model for sentiment analysis. 

<center><img src="AndriodApp/sentiment.gif" width="75%"/></center>


## 4. Results & Observations

For each of 6 GLUE tasks, we run 144 different heads pruning and 9 different Layer Dropping experiments, then evaluate performance base on their [GLUE metric](https://gluebenchmark.com/tasks). The experiment design flow to obtain the result in this section is as below:

<center><img src="logs/plots/design_flow.png" width="75%"/></center>

In 4., we will analyze the result and share our observations.

### 4.1 Effect of Heads Pruning

In this section, we complete our experiments result with the paper [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) for which we reproduced their methods.


#### 4.1.1 Effect of Ablating One Head

In the original paper, researchers only presents result of WMT model(in Table 1) and leads to conclusion that: at test time, most heads are redundant given the rest of the model. Here we will complete those experiments and see whether they are valid statements.

- Heads hurts performance in most cases:

  <center><img src="logs/plots/head_prune_accuracy.png" width="75%"/></center>

   We notice that the effect of attention heads pruning are different for each task. the red dot line in this plot is the accuracy of the full model without any pruning. we notice that for task like SST-2 and MRPC, pruning nearly half of attention heads actually result in a *higher* accuracy score! This indicates that at test time, most heads are redundant. However, For tasks like RTE and STS-B, pruning most of the heads are harmful for accuracy. What is the reason behind this phenomenon? We will discuss about it later.


- But the performance drops are very task-specific!
  <details open>
  <summary>
  📈 <strong>MRPC</strong>
  </summary>
  <img src="logs/plots/head_prune_mrpc_accuracy.jpg"" />
  </details>

  <br>

  <details open>
  <summary>
  📈 <strong>SST-2</strong>
  </summary>
  <img src="logs/plots/head_prune_sst2_accuracy.jpg" />
  </details>

  <br>

  <details>
  <summary>
  📉 <strong>WNLI</strong>
  </summary>
  <img src="logs/plots/head_prune_wnli_accuracy.jpg" />
  </details>

  <br>

  <details>
  <summary>
  📉 <strong>COLA</strong>
  </summary>
  <img src="logs/plots/head_prune_cola_accuracy.jpg" />
  </details>

  <br>

  <details>
  <summary>
  📉 <strong>STS-B</strong>
  </summary>
  <img src="logs/plots/head_prune_stsb_accuracy.jpg" />
  </details>

  <br>

  <details>
  <summary>
  📉 <strong>RTE</strong>
  </summary>
  <img src="logs/plots/head_prune_rte_accuracy.jpg" />
  </details>

  **Conclusion from above tables: The performance head-pruning is related to the difficulty of tasks.** 

  Let's first think about a question, "What makes it possible for pruning"? We think it relies on the **redundancy in parameters**. Especially for those tasks that are relatively "easy", we don't need all the heads to perform equally well on our downstream tasks. For example, for SST-2 task, we notice that pruning all the attention heads actually result in a *higher* or almost same accuracy score. Why is that? We believe it's because the SST-2 task is just a sentiment binary classification task, therefore we don't even need self-attention to capture the **context** information in order to perform well on this task. After all, before the prevalence of neural networks, people even used **bag-of-words** and simple logistic regression to perform binary sentiment classification! We know that bag-of-words completely disregard the order of words, which indicates that binary sentiment classification task can perform well even without the order of words. 

  On the other hand, *let's think about what makes BERT powerful*. BERT firstly uses WordPiece embeddings as input, and this embedding is meant to learn *context-independent* representations, whereas the transformer layer with self-attention are meant to learn *context-dependent* representations. However, the power of BERT-like representations comes mostly from the use of context-dependent representations. Without these context-dependent representations enabled by self-attention mechanism, BERT is nothing more than traditional Word2vec. 

  Let's think about the sentiment binary classification task again. Since we can even perform well with bag-of-words representations, this means that we can already perform well with context-independent representations. Therefore, most attention heads that captures context-dependent information are, to some extent, redundant. This is probably why we can prune most of the attention heads without resulting in noticeable accuracy drop. 

  However, we notice that for RTE(The Recognizing Textual Entailment datasets) task, pruning most of the attention heads results in worst result. In order to understand the reason behind this phenomenon, we need to know what RTE task is about. Well, RTE is a classification task that identifies whether the relationship between two sentences are *neutral, contradiction, or entailment*. This is a challenging logical problem that is even hard for human, and apparently the order of words is very important ("A entails B" is not equal to "B entails A").  Therefore, we need to rely on BERT to capture context information with self-attention mechanism, and every attention heads are indispensable. 


#### 4.1.2 Are heads universally important?

In the NIPS paper,  the researchers conclude that the important heads are “universally” important (By Figure 2), however, they reach this conclusion only by using two “out-of-domain” test set: for the original Transformer model, they use newstest2013(English-German data) and MTNT(English-French data); for BERT, they use MNLI mismatched validation set and MNLI matched set. ）These two “out-of-domain” test set, from our perspective, are not totally unrelated.

Therefore, we perform more experiments on six intrinsically different tasks, and  we can see correlation for each task pair in this scatter plot:

<center><img src="logs/plots/head_prune_correlation.png" width="75%"/></center>

- We notice that although some task pairs have high correlation scores, there are still many task pairs with correlation scores close to zero, and some correlation scores are even negative, suggesting that deleting same attention heads may have opposite effect on two different tasks. Therefore, our extensive experiment seems to disprove the conclusion in that NIPS paper -- we found that the importance of heads are task-dependent rather than "universal".


### 4.2 Layer drop

Our work builds on similar observations of heads pruning. On one hand, we recognize the need of finetuning the model again after we simplify the model structure. On the other hand, our goal is to deploy BERT in edge device so we further question whether it is necessary to use all layers of a pre-trained model in downstream tasks! 

Let's look at the result of layer pruning:

- In some task droping 2 layers result in acceptable performance decrease.

  <center><img src="logs/plots/layer_drop_acc_comparison.png" width="75%"/></center>
  
  We notice that top layer pruning and symmetric layer pruning result in less performance drop, while bottom layer pruning result in large performance drop. This is probably because  bottom layer captures the low-level word representation that is important for upper layers. We use the strategy to balance latency-accuracy tradeoff: for strategy results in tolerable performance drop,we pick the one that drop most layers.



- And two of the tasks, RTE and WNLI show significant time cost improved. 

<center><img src="logs/plots/layer_drop_time_comparison.png" width="50%"/></center>

  Layers Drop reduces significant amount of fine-tuning & inference time, and also reduce parameter size greatly.
  
  Also, as we successfully deploy the 6-layers DistilBert that has same structure as 6 layers dropped BERT, we can verify the idea of layer dopping is deployable. Also, compared with DistilBERT, in our method we apply layer drop to pretrained BERT so we only need finetuning time. 


### 4.3 Observations / Conclusions

The idea behind pruning is that deep neural models are overparameterized and that not all strictly needed especially during inference. 

We conclude that: 

1. The performance of head-pruning is related to the difficulty of tasks.

    - In simple task like sentiment classification, some heads are similar so that we can safely drop without affecting performance.

    - In difficult task that requires contextual understanding, each head are indispensable. 

    - the result in the NIPS paper is flawed – attention heads are not universally important across different tasks. The lesson is, Strong argument requires complete experiments!

2. layer drop is a good strategy to use under resource limitations. It can achieve a 46% reduction in inference time, 40% reduction in parameter size while maintaining 90%+ of the original score.

