1，安装

pip install pytorch

some code is from https://github.com/GeneZC/ASGCN, and thanks them

pip install transformers

pip install spacy 

install resources for spacy

run nlp = spacy.load('en_core_web_sm') to check

or you can pip install stanfordcorenlp

and install stanford-corenlp-full-2018-02-27 for stanfordcorenlp

use nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27') to check

2，下载数据集并生成依赖文件

copy datasets from https://github.com/GeneZC/ASGCN 

use the path and change the corresponding parameter in following files

run generateGraph_spacy.py or generateGraph_stanford.py to generate adjacent matrix file for all five datasets

run data_utils.py to generate pkl form of all five datasets

3，训练与测试

change some parameters in following files including path, dataset name and so on

run train.py to train and evaluate, the first time you run this code it will download BERT-related files from outer website, please wait for it to complete the downloading.

4，自选模块与微调

默认为双Transformer模型（Transformer+GCN），部分注释的内容为可选模块，包含掩码自注意力，双向GCN等，以及其他可调超参
