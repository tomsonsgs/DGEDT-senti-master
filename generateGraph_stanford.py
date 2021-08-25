# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import tqdm
#nlp = spacy.load('en_core_web_sm')
import re
from stanfordcorenlp import StanfordCoreNLP
import json
nlps = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27')
#text = 'Guangdong University of Foreign  Studies is located in Guangzhou. Guangdong University of Foreign Studies'
#print ('Tokenize:', nlp.word_tokenize(text))
def tokenizeanddep(text):
    text+=' '
    text=re.sub(r'\. ',' . ',text).strip()
#    print(text)
    text=re.sub(r' {2,}',' ',text)
    nlp_properties = {
                'annotators': "depparse",
    #            "tokenize.options": "splitHyphenated=false,normalizeParentheses=false",
                "tokenize.whitespace": True,  # all tokens have been tokenized before
                'ssplit.isOneSentence': False,
                'outputFormat': 'json'
            }
    try:
        parsed = json.loads(nlps.annotate(text.strip(), nlp_properties))
    except:
        print('ewewerror')
    parsed=parsed['sentences']
    tokens=[]
    tuples=[]
    tmplen=0
    for item in parsed:
        tokens.extend([ite['word'] for ite in item['tokens']])
        
        tuples.extend( [(ite['dep'],ite['governor']-1+tmplen,ite['dependent']-1+tmplen) for ite in item['basicDependencies'] if ite['dep']!='ROOT'])
        tmplen=len(tokens)
    return tokens,tuples
#a=tokenizeanddep(text)
#nlp.close()
#def tokenize(text):
#    text=text.strip()
#    text=re.sub(r' {2,}',' ',text)
#    document = nlp(text)
#    return [token.text for token in document]
def tokenize(text):
    text=text.strip()
    text=re.sub(r' {2,}',' ',text)
    document,_ = tokenizeanddep(text)
    return document
def update_edge(text,vocab):
    # https://spacy.io/docs/usage/processing-text
    _,document = tokenizeanddep(text)
#    seq_len = len(text.split())
    for token in document:
           if token[0] not in vocab:
               vocab[token[0]]=len(vocab)
    return 0
def dependency_adj_matrix(text,edge_vocab):
    # https://spacy.io/docs/usage/processing-text
    tokens,document = tokenizeanddep(text)
    seq_len = len(tokens)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    matrix1 = np.zeros((seq_len, seq_len)).astype('float32')
    edge = np.zeros((seq_len, seq_len)).astype('int32')
    edge1 = np.zeros((seq_len, seq_len)).astype('int32')
    for i in range(seq_len):
        matrix[i,i]=1
        matrix1[i,i]=1
#        edge[i,i]=1
#        edge1[i,i]=1
#    print(' '.join(tokens))
#    print(document)
#    a=input('hahha')
#    assert max([item[1] for item in document]+[item[2] for item in document])<=seq_len-1
    for span in document:
#        if token.i >= seq_len:
#            print('bug')
#            print(text)
#            print(text.split())
#            print(document)
#            print([token.i for token in document])
#            print([token.text for token in document])
#            a=input('hahha')
#        if token.i < seq_len:
            matrix[span[1]][span[2]] = 1
            matrix1[span[2]][span[1]] = 1
            # https://spacy.io/docs/api/token
#            for child in token.children:
#                if child.i < seq_len:
#                    matrix[token.i][child.i] = 1
#                    matrix1[child.i][token.i] = 1
            edge[span[1]][span[2]] = edge_vocab.get(span[0],1)
            edge1[span[2]][span[1]] = edge_vocab.get(span[0],1)
#    print(matrix,edge)
#    a=input('hahha')
    return matrix,matrix1,edge,edge1
def concat(texts,aspect):
    source=''
    splitnum=0
    haha=[]
    for i,text in enumerate(texts):
        source+=text
        splitnum+=len(tokenize(text))
        haha.extend(tokenize(text))
        if i <len(texts)-1:
           source+=' '+aspect+' '
           splitnum+=len(tokenize(aspect))
    if splitnum!=len(tokenize(source.strip())):
        print(haha)
        print(texts)
        print(aspect)
        print(source)
        print(splitnum)
        print(tokenize(source.strip()))
        print(len(tokenize(source.strip())))
        a=input('gfg')
    return re.sub(r' {2,}',' ',source.strip())
def process(filename,edge_vocab=None,savevocab=True):
    if edge_vocab is not None:
        pass
    else:
        edge_vocab={'<pad>':0,'<unk>':1}
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    if savevocab:
        fout1 = open(filename+'.edgevocab', 'wb')
    if savevocab:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            text_left = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            update_edge(concat(text_left,aspect),edge_vocab)
    for i in tqdm.tqdm(range(0, len(lines), 3)):
        text_left = [s.lower().strip() for s in lines[i].split("$T$")]
        aspect = lines[i + 1].lower().strip()
#        print(lines[i])
#        concat(text_left,aspect)
        adj_matrix = dependency_adj_matrix(concat(text_left,aspect),edge_vocab)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout) 
    if savevocab:
        pickle.dump(edge_vocab, fout1)       
    fout.close() 
    if savevocab:
        fout1.close() 
    return edge_vocab

if __name__ == '__main__':
    edge_vocab=process('./datasets/acl-14-short-data/train.raw',None,True)
    process('./datasets/acl-14-short-data/test.raw',edge_vocab,False)
    edge_vocab=process('./datasets/semeval14/restaurant_train.raw',None,True)
    process('./datasets/semeval14/restaurant_test.raw',edge_vocab,False)
    edge_vocab=process('./datasets/semeval14/laptop_train.raw',None,True)
    process('./datasets/semeval14/laptop_test.raw',edge_vocab,False)
    edge_vocab=process('./datasets/semeval15/restaurant_train.raw',None,True)
    process('./datasets/semeval15/restaurant_test.raw',edge_vocab,False)
    edge_vocab=process('./datasets/semeval16/restaurant_train.raw',None,True)
    process('./datasets/semeval16/restaurant_test.raw',edge_vocab,False)
    nlps.close()