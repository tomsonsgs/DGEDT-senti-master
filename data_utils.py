# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from generateGraph_spacy import tokenize,concat
import tqdm
import re
from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW, WarmupLinearSchedule
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer.
#bert_model= BertModel.from_pretrained('bert-base-uncased')
def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
#    if os.path.exists(embedding_matrix_file_name):
#        print('loading embedding_matrix:', embedding_matrix_file_name)
#        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
#    else:
    if True:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None,tokenizer=None):
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.do_basic_tokenize=False
        print('load successfully')
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower().strip()
        words = tokenize(text)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, tran=False):
        text = text.lower().strip()
        words = tokenize(text)
        trans=[]
        realwords=[]

        for word in words:
            wordpieces=self.tokenizer.tokenize(word)
            tmplen=len(realwords)
            
            realwords.extend(wordpieces)
            trans.append([tmplen,len(realwords)])
#        unknownidx = 1_convert_token_to_id
        print(realwords)
        sequence = [self.tokenizer._convert_token_to_id('[CLS]')]+[self.tokenizer._convert_token_to_id(w) for w in realwords]+[self.tokenizer._convert_token_to_id('[SEP]')]
        if len(sequence) == 0:
            sequence = [0]
        if tran: return sequence,trans
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def span(texts,aspect):
    startid=0
    aslen=len(tokenize(aspect))
    spans=[]
    for idx,text in enumerate(texts):
        tmp=len(tokenize(text))
        startid+=tmp
        tmp=startid
        if idx < len(texts)-1:
            startid+=aslen
            spans.append([tmp,startid])
    return spans
class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left = [s.lower().strip() for s in lines[i].split("$T$")]
                aspect = lines[i + 1].lower().strip()
#                update_edge(concat(text_left,aspect),edge_vocab)
#                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
#                aspect = lines[i + 1].lower().strip()
                text_raw = concat(text_left,aspect)
                text += text_raw + " "
        print(text)
        return text

    @staticmethod
    def __read_data__(fname, tokenizer,fname1):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        fin1 = open(fname1+'.edgevocab', 'rb')
        idx2gragh = pickle.load(fin)
        edgevocab=pickle.load(fin1)
        fin1.close()
        fin.close()


        all_data = []
        for i in tqdm.tqdm(range(0, len(lines), 3)):
#            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
#            aspect = lines[i + 1].lower().strip()
            text_left = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            span_indices=span(text_left,aspect)
            assert len(span_indices)>=1
            concats=concat(text_left,aspect)
            text_indices, tran_indices = tokenizer.text_to_sequence(concats,True)
            context_indices = tokenizer.text_to_sequence(concats)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(concats)
            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            assert len(idx2gragh[i][0])==len(tokenize(concats))
           # print(tokenize(concats))
            print(span_indices)
            print(tran_indices)
            print(aspect)
            print(dependency_graph[0])
            a=input('fdfdf')
            data = {
                'text': tokenize(concats.lower().strip()),
                'aspect': tokenize(aspect),
                'text_indices': text_indices,
                'tran_indices': tran_indices,
                'context_indices': context_indices,
                'span_indices': span_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
            }

            all_data.append(data)
        return all_data
    def __read_datas__(fname, tokenizer,fname1):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname1, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines1 = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        fin1 = open(fname+'.edgevocab', 'rb')
        idx2gragh = pickle.load(fin)
        edgevocab=pickle.load(fin1)
        fin1.close()
        fin.close()


        all_data = []
        for i in tqdm.tqdm(range(0, len(lines), 1)):
#            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
#            aspect = lines[i + 1].lower().strip()
#            text_left = [s.lower().strip() for s in lines[i].split("$T$")]
#            aspect = lines[i + 1].lower().strip()
            pola = lines1[i].strip()
#            span_indices=span(text_left,aspect)
#            assert len(span_indices)>=1
            concats=re.sub(r' {2,}',' ',lines[i].lower().strip())
            text_indices, tran_indices = tokenizer.text_to_sequence(concats,True)
#            context_indices = tokenizer.text_to_sequence(concats)
#            aspect_indices = tokenizer.text_to_sequence(aspect)
#            left_indices = tokenizer.text_to_sequence(concats)
            if pola=='negative':
                polarity=0
            elif pola=='neutral':
                    polarity=1
            else:polarity=2
                
            dependency_graph = idx2gragh[i]
            assert len(idx2gragh[i][0])==len(tokenize(concats))
#            print(tokenize(concats))
#            print(span_indices)
#            print(aspect)
#            a=input('fdfdf')
            data = {
                'text': tokenize(concats.lower().strip()),
                'aspect': None,
                'text_indices': text_indices,
                'tran_indices': tran_indices,
                'context_indices': None,
                'span_indices': None,
                'aspect_indices': None,
                'left_indices': None,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
            }

            all_data.append(data)
        return all_data
    def __init__(self, dataset='rest14', embed_dim=300,change=False):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
            'other': {
                'train': './datasets/datas/amazon_review.txt',
                'test': './datasets/datas/amazon_label.txt'
            }

        }
#        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        self.tokenizer = Tokenizer()
#        if os.path.exists(dataset+'_word2idx.pkl'):
#            print("loading {0} tokenizer...".format(dataset))
#            with open(dataset+'_word2idx.pkl', 'rb') as f:
#                 word2idx = pickle.load(f)
#                 tokenizer = Tokenizer(word2idx=word2idx)
#        else:
#            tokenizer = Tokenizer()
#            tokenizer.fit_on_text(text)
#            with open(dataset+'_word2idx.pkl', 'wb') as f:
#                 pickle.dump(tokenizer.word2idx, f)
        self.edgevocab=pickle.load(open(fname[dataset]['train']+'.edgevocab', 'rb'))
        self.embedding_matrix = build_embedding_matrix(self.tokenizer.tokenizer.vocab, embed_dim, dataset)
        if change:
            self.train_data = ABSADataset(ABSADatesetReader.__read_datas__(fname[dataset]['train'], self.tokenizer,fname[dataset]['test']))
        else:
            self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], self.tokenizer,fname[dataset]['train']))
            self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], self.tokenizer,fname[dataset]['train']))
if __name__ == '__main__':
    tmp=ABSADatesetReader(dataset='twitter')
    dataset='twitter'
    with open(dataset+'_datas.pkl', 'wb') as f:
                     pickle.dump(tmp, f)
    tmp=ABSADatesetReader(dataset='rest14')
    dataset='rest14'
    with open(dataset+'_datas.pkl', 'wb') as f:
                     pickle.dump(tmp, f)
    tmp=ABSADatesetReader(dataset='lap14')
    dataset='lap14'
    with open(dataset+'_datas.pkl', 'wb') as f:
                     pickle.dump(tmp, f)
    tmp=ABSADatesetReader(dataset='rest15')
    dataset='rest15'
    with open(dataset+'_datas.pkl', 'wb') as f:
                     pickle.dump(tmp, f)
    tmp=ABSADatesetReader(dataset='rest16')
    dataset='rest16'
    with open(dataset+'_datas.pkl', 'wb') as f:
                     pickle.dump(tmp, f)
#    tmp=ABSADatesetReader(dataset='other',change=True)
#    dataset='other'
#    with open(dataset+'_datas.pkl', 'wb') as f:
#                     pickle.dump(tmp, f)
#tmp=pickle.load(open(dataset+'_datas.pkl', 'rb'))
    