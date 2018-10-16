# Text matching on Quora qestion-answer pair dataset

## Introduction

### a brief review of the Quora Question Pair (QQP) Task

[Quora Pair Dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) is a dataset of 400,000 question pairs from the [Quora forum](https://www.quora.com/), where people raise questions for the others to answer. Each sample in the dataset consists of two English questions and a label represent whether the two questions are duplicate. The dataset is well annotated by human. 

Below are two samples of the dataset. The last clolmn indicates whether the two questions are duplicate (1) or not(0).

|id | qid1 | qid2| question1| question2| is_duplicate
|:---:|:---:|:---:|:---:|:---:|:---:|
|0 |1 |2 |What is the step by step guide to invest in share market in india? |What is the step by step guide to invest in share market? |0|
|1 |3 |4 |What is the story of Kohinoor (Koh-i-Noor) Diamond? | What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back? |0|

 A [kaggle competition](https://www.kaggle.com/c/quora-question-pairs#description) is held base on this dataset in 2017. The kaggler is able to reach the train_data(with label) and the test_data(without label), and is requested to make predictions on the test_data. The predictions are evaluated by the log-likelihood loss on the test_data.

The kaggle competition has inspired lots of effective work. However, most of the models are rule-based, thus are hard to transfer to new tasks. Researchers keep seeking for more general models that works well on this task and the other NLP(Natual Language Processing) tasks.

[Wang et al.](https://arxiv.org/abs/1702.03814) proposed the BIMPM(Bilateral Multi-Perspective Matching) model based on the Quora Question Pair dataset. They splited the original dataset to [3 part](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing): train.tsv(384,348 samples), dev.tsv(10,000 samples) and test.tsv(10,000 samples). The class distribution in train.tsv is unbalanced(37% positive, 63% negative). But the class distribution in dev.tsv and test.tsv is balanced(50% positive and 50% negetive). We follow this split in our experiments. 

### Our Work

Based on the Quora Question Pair Dataset, we will implement some classic models in the area of the NLU(Neraul Lanuage Understanding). The prediction results will be evaluated by accuracy on the test.tsv, like [Wang et al.](https://arxiv.org/abs/1702.03814).

## Environment Preparation: Install fluid release 1.0

You can follow the fluid's [official document](http://www.paddlepaddle.org/documentation/docs/en/1.0/build_and_install/pip_install_en.html) to install the fluid. 

[Attention] You are supposed to install python and pip before installing fluid

### cpu version

```
pip install paddlepaddle==1.0.1
```

### gpu version

Assume you have downloaded cuda(cuda9.0) and cudnn(cudnn7) lib, here is an expample:

```shell

pip install paddlepaddle-gpu==1.0.1.post97

```

### Have I installed Fluid successfully?

You can run the following script in your command line:

```shell
python -c "import paddle"
```

Fluid is installed successfully if no error message is prompted. If you get any error, feel free to open issues under the [PaddlePaddle repository](https://github.com/PaddlePaddle/Paddle/issues). 

## Prepare Data

Please download the Quora dataset firstly from https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
 to ROOT_DIR $HOME/.cache/paddle/dataset

Then run the data/prepare_quora_data.sh to download the pretrained embedding glove.840B.300d.zip:

```shell
cd data
sh prepare_quora_data.sh   
```

The finally dataset dir should be like

```shell

$HOME/.cache/paddle/dataset
    |- Quora_question_pair_partition
        |- train.tsv
        |- test.tsv
        |- dev.tsv
        |- readme.txt
        |- wordvec.txt
    |- glove.840B.300d.txt
```


### Train

```shell
fluid train_and_evaluate.py  \
    --model_name=cdssmNet  \
    --config=cdssm_base
```

应该得到这样的log:

```shell
Generating word dict...
('Vocab size: ', 31966)
loading word2vec from  data/glove.840B.300d.txt
preparing pretrained word embedding ...
pretrained_word_embedding to be load: [[ 2.7204001e-01 -6.2029999e-02 -1.8840000e-01 ...  1.3015001e-01
  -1.8317001e-01  1.3230000e-01]
 [-3.8548000e-02  5.4251999e-01 -2.1843000e-01 ...  1.1798000e-01
   2.4590001e-01  2.2872999e-01]
 [-8.4960997e-02  5.0199997e-01  2.3823001e-03 ... -2.1511000e-01
  -2.6304001e-01 -6.0172998e-03]
 ...
 [ 1.5450897e-02 -7.1158134e-03  3.1940900e-02 ... -3.0432804e-02
  -2.8658450e-02 -4.9563847e-02]
 [-3.0259702e-02 -2.0112008e-02 -1.5854578e-03 ... -4.8602581e-02
  -2.7053220e-02 -2.1766458e-02]
 [ 3.3008438e-04  3.1509984e-03  4.1116238e-02 ... -1.5571933e-02
   1.0166970e-02 -5.9445472e-03]]
net_name:  cdssm
net_config:  {'emb_dim': 300, 'fc_hid_dim': 30, 'dict_dim': 31966, 'kernel_count': 64, 'kernel_size': 5}
global_config:  {'learning_rate': 0.001, 'save_dirname': 'cdssm_model', 'learing_rate': 0.001, 'embedding_norm': False, 'train_samples_num': 323423, 'epoch_num': 30, 'use_pretrained_word_embedding': True, 'lr_decay': 1, 'batch_size': 128}
config: {'kernel_count': 64, 'emb_dim': 300, 'dict_dim': 31966, 'drop_rate': 0.2, 'fc_hid_dim': 30, 'kernel_size': 5}
param name: emb.w; param shape: (31966L, 300L)
param name: conv1d.w; param shape: (1500L, 64L)
param name: fc1.w; param shape: (64L, 30L)
param name: fc1.b; param shape: (30L,)
param name: fc_2.w_0; param shape: (60L, 2L)
param name: fc_2.b_0; param shape: (2L,)
loading pretrained word embedding to param
epoch_id: 0, batch_id: 0, acc: 0.484375, cost: 0.763859
epoch_id: 0, batch_id: 100, acc: 0.609375, cost: 0.638024
epoch_id: 0, batch_id: 200, acc: 0.734375, cost: 0.534823
epoch_id: 0, batch_id: 300, acc: 0.812500, cost: 0.493679
epoch_id: 0, batch_id: 400, acc: 0.789062, cost: 0.464649
epoch_id: 0, batch_id: 500, acc: 0.718750, cost: 0.527808
epoch_id: 0, batch_id: 600, acc: 0.789062, cost: 0.493155
epoch_id: 0, batch_id: 700, acc: 0.750000, cost: 0.500162
epoch_id: 0, batch_id: 800, acc: 0.773438, cost: 0.476200
epoch_id: 0, batch_id: 900, acc: 0.820312, cost: 0.484623
epoch_id: 0, batch_id: 1000, acc: 0.765625, cost: 0.496225
epoch_id: 0, batch_id: 1100, acc: 0.742188, cost: 0.548540
epoch_id: 0, batch_id: 1200, acc: 0.742188, cost: 0.568234
epoch_id: 0, batch_id: 1300, acc: 0.742188, cost: 0.489356
epoch_id: 0, batch_id: 1400, acc: 0.804688, cost: 0.451339
epoch_id: 0, batch_id: 1500, acc: 0.679688, cost: 0.587591
epoch_id: 0, batch_id: 1600, acc: 0.812500, cost: 0.426993
epoch_id: 0, batch_id: 1700, acc: 0.687500, cost: 0.547271
epoch_id: 0, batch_id: 1800, acc: 0.796875, cost: 0.489567
epoch_id: 0, batch_id: 1900, acc: 0.804688, cost: 0.436827
epoch_id: 0, batch_id: 2000, acc: 0.765625, cost: 0.468468
epoch_id: 0, batch_id: 2100, acc: 0.718750, cost: 0.502783
epoch_id: 0, batch_id: 2200, acc: 0.726562, cost: 0.461317
epoch_id: 0, batch_id: 2300, acc: 0.710938, cost: 0.587220
epoch_id: 0, batch_id: 2400, acc: 0.750000, cost: 0.553239
epoch_id: 0, batch_id: 2500, acc: 0.781250, cost: 0.488242

epoch_id: 0, train_avg_acc: 0.754466, train_avg_cost: 0.505327
epoch_id: 0, dev_acc: 0.778795, dev_cost: 0.468318
epoch_id: 0, test_acc: 0.776339, test_cost: 0.473108

epoch_id: 1, batch_id: 0, acc: 0.757812, cost: 0.394520
epoch_id: 1, batch_id: 100, acc: 0.835938, cost: 0.423348
epoch_id: 1, batch_id: 200, acc: 0.859375, cost: 0.408528
epoch_id: 1, batch_id: 300, acc: 0.757812, cost: 0.472944
epoch_id: 1, batch_id: 400, acc: 0.812500, cost: 0.422836
epoch_id: 1, batch_id: 500, acc: 0.703125, cost: 0.604955
epoch_id: 1, batch_id: 600, acc: 0.796875, cost: 0.433807
epoch_id: 1, batch_id: 700, acc: 0.843750, cost: 0.367446
epoch_id: 1, batch_id: 800, acc: 0.828125, cost: 0.403060
epoch_id: 1, batch_id: 900, acc: 0.796875, cost: 0.462563
epoch_id: 1, batch_id: 1000, acc: 0.812500, cost: 0.439250
epoch_id: 1, batch_id: 1100, acc: 0.820312, cost: 0.374529
epoch_id: 1, batch_id: 1200, acc: 0.828125, cost: 0.451674
epoch_id: 1, batch_id: 1300, acc: 0.820312, cost: 0.431569
epoch_id: 1, batch_id: 1400, acc: 0.804688, cost: 0.406776
epoch_id: 1, batch_id: 1500, acc: 0.843750, cost: 0.383016
epoch_id: 1, batch_id: 1600, acc: 0.804688, cost: 0.383064
epoch_id: 1, batch_id: 1700, acc: 0.796875, cost: 0.434245
epoch_id: 1, batch_id: 1800, acc: 0.789062, cost: 0.467287
epoch_id: 1, batch_id: 1900, acc: 0.789062, cost: 0.381877
epoch_id: 1, batch_id: 2000, acc: 0.773438, cost: 0.535949
epoch_id: 1, batch_id: 2100, acc: 0.804688, cost: 0.446306
epoch_id: 1, batch_id: 2200, acc: 0.867188, cost: 0.393089
epoch_id: 1, batch_id: 2300, acc: 0.828125, cost: 0.388044
epoch_id: 1, batch_id: 2400, acc: 0.812500, cost: 0.435385
epoch_id: 1, batch_id: 2500, acc: 0.835938, cost: 0.356312

epoch_id: 1, train_avg_acc: 0.799810, train_avg_cost: 0.429873
epoch_id: 1, dev_acc: 0.792361, dev_cost: 0.458000
epoch_id: 1, test_acc: 0.787376, test_cost: 0.461890

```

||
