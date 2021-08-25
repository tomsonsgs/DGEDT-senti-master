# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, other=None, sort_key='text_indices', shuffle=True, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batch_size=batch_size
        self.data=[item for item in data if len(item['text_indices'])<100 ]
        if other is not None:
            self.other=[item for item in other if len(item['text_indices'])<100 ]
            random.shuffle(self.other)
            self.batches = self.sort_and_pad(data, batch_size, self.other)
#            self.others=self.sort_and_pad([], batch_size, self.other)
        else:
            self.other=None
            self.batches = self.sort_and_pad(self.data, batch_size)
        
        self.batch_len = len(self.batches)
#    def combine(self):
#        random.shuffle(self.other)
#        for i in range(self.batch_len):
#            batches[i]
    def sort_and_pad(self, data, batch_size, other=None):
#        data=[item for item in data if len(item['text_indices'])<100 ]
        num_batch = int(math.ceil(len(data) / batch_size))

#        if self.sort:
#            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
#        else:
        sorted_data = data
        if other is not None:
            num_k = int(math.ceil(len(other) / batch_size))    
            batches = []
            for i in range(num_batch):
                if i<num_k: k=i
                else:
                    k=random.randint(0,num_k-1)
                batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size],other[k*batch_size : (k+1)*batch_size]))
        else:
            batches = []
            for i in range(num_batch):
                batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data, other_data=None):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_dependency_graph1 = []
        batch_dependency_graph2 = []
        batch_dependency_graph3 = []
        batch_span=[]
        batch_tran=[]
        batch_text=[]
        batch_aspect=[]
        if other_data is not None:
            max_len = max([len(t[self.sort_key]) for t in batch_data+other_data])
            max_len1 = max([len(t['tran_indices']) for t in batch_data+other_data])
        else:
            max_len = max([len(t[self.sort_key]) for t in batch_data])
            max_len1 = max([len(t['tran_indices']) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, span_indices,tran_indices,aspect_indices, left_indices, polarity, dependency_graph,text,aspect = \
                item['text_indices'], item['context_indices'],item['span_indices'],item['tran_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'],item['text'], item['aspect']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_span.append(span_indices)
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_tran.append(tran_indices)
            batch_dependency_graph.append(numpy.pad(dependency_graph[0], \
                ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
            batch_dependency_graph1.append(numpy.pad(dependency_graph[1], \
                ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
            batch_dependency_graph2.append(numpy.pad(dependency_graph[2], \
                ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
            batch_dependency_graph3.append(numpy.pad(dependency_graph[3], \
                ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
            batch_text.append(text)
            batch_aspect.append(aspect)
        if other_data is not None:
            for item in other_data:
                text_indices, context_indices, span_indices,tran_indices,aspect_indices, left_indices, polarity, dependency_graph,text,aspect = \
                    item['text_indices'], item['context_indices'],item['span_indices'],item['tran_indices'], item['aspect_indices'], item['left_indices'],\
                    item['polarity'], item['dependency_graph'],item['text'], item['aspect']
                text_padding = [0] * (max_len - len(text_indices))
                batch_text_indices.append(text_indices + text_padding)
                batch_text.append(text)
                batch_polarity.append(polarity)
                batch_tran.append(tran_indices)
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'text':batch_text,\
                'aspect':batch_aspect,\
                'context_indices': torch.tensor(batch_context_indices), \
                'span_indices':batch_span,\
                'tran_indices':batch_tran,\
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph),\
                'dependency_graph1': torch.tensor(batch_dependency_graph1),\
                'dependency_graph2': torch.tensor(batch_dependency_graph2).long(),\
                'dependency_graph3': torch.tensor(batch_dependency_graph3).long()
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            self.batches = self.sort_and_pad(self.data, self.batch_size)
        for idx in range(self.batch_len):
#            print(len(self.batches[idx]['text_indices']))
            yield self.batches[idx]
