import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


import copy
def length2mask(length,maxlength):
    size=list(length.size())
    length=length.unsqueeze(-1).repeat(*([1]*len(size)+[maxlength])).long()
    ran=torch.arange(maxlength).cuda()
    ran=ran.expand_as(length)
    mask=ran<length
    return mask.float().cuda()
class selfatt(nn.Module):
    def __init__(self,hidden_size):
        """Base RNN Encoder Class"""
        super(selfatt, self).__init__()
        self.linear1=torch.nn.Linear(2*hidden_size,hidden_size)
        self.linear2=torch.nn.Linear(hidden_size,1)
    def forward(self,inputs,final_state,mask):#b,s,h;b,h;b,s
        final_state=final_state.unsqueeze(1).expand_as(inputs)
        fuse=torch.cat([inputs,final_state],-1)
        fuse=self.linear2(torch.tanh(self.linear1(fuse))).squeeze(-1)#b,s
        mask=(1-mask)*-1e10
        fuse=fuse+mask
        att=torch.softmax(fuse,-1)#b,s
        fuse=torch.matmul(att.unsqueeze(1),inputs).squeeze(1)#b,h
        return fuse,att
class mulatt(nn.Module):
    def __init__(self,hidden_size):
        """Base RNN Encoder Class"""
        super(mulatt, self).__init__()
        self.linear1=torch.nn.Linear(hidden_size,hidden_size,bias=False)
#        self.linear2=torch.nn.Linear(hidden_size,1)
    def forward(self,inputs,mask,selfatt):#b,c,s,h;b,c,s;b,c,s
#        print(inputs.size())
#        print(mask.size())
#        print(selfatt.size())
        context_size=inputs.size(1)
        sentence_size=inputs.size(2)
        inputs1=inputs.unsqueeze(1).repeat(1,context_size,1,1,1)#b,c2,c1m,s1,h
        mask=mask.unsqueeze(1).repeat(1,context_size,1,1)#b,c2,c1m,s1
        mask=mask.unsqueeze(-2).repeat(1,1,1,sentence_size,1)#b,c2,c1m,s2,s1
        mask=(1-mask)*-1e10
        inputs2=inputs.unsqueeze(2).repeat(1,1,context_size,1,1)#b,c2m,c1,s2,h, this is the base
        selfatt=selfatt.unsqueeze(2).repeat(1,1,context_size,1).unsqueeze(-2)#b,c2m,c1,1,s2
        inputs1=inputs1.transpose(3,4)
        fused=torch.matmul(self.linear1(inputs2),inputs1)+mask#b,c2,c1,s2,s1
        att=torch.softmax(fused,-1)#b,c2,c1,s2,s1
        att=torch.sigmoid(fused)#b,c2,c1,s2,s1
#        print(att.size())
#        print(inputs1.size())
        fused=torch.matmul(att,inputs1.transpose(3,4))#b,c2,c1,s2,h
        fused=torch.matmul(selfatt,fused).squeeze(-2)#b,c2m,c1,h
        return fused,att
def to_var(x, on_cpu=False, gpu_id=None, dasync=False):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, dasync)
        #x = Variable(x)
    return x
class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size)),
                    to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size)))
        else:
            return to_var(torch.zeros(self.num_layers*self.num_directions,
                                        batch_size,
                                        self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnn=nn.LSTM, num_layers=1, bidirectional=True,
                 dropout=0.0, bias=True, batch_first=True):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.rnn = rnn(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_seq_len]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.data.tolist()

        # [num_sentences, max_source_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_sentences, max_source_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(embedded, input_length_sorted,
                                            batch_first=self.batch_first)

        hidden = self.init_h(batch_size, hidden=hidden)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first,total_length=seq_len)
        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                        hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden

class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.LSTM, num_layers=1, dropout=0.0,
                 bidirectional=True, bias=True, batch_first=True):
        """Context-level Encoder"""
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size,
                        hidden_size=context_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_length, hidden=None):
        """
        Args:
            encoder_hidden (Variable, FloatTensor): [batch_size, max_len, num_layers * direction * hidden_size]
            conversation_length (Variable, LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_seq_len, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len, _  = encoder_hidden.size()

        # Sort for PackedSequence
        conv_length_sorted, indices = conversation_length.sort(descending=True)
        conv_length_sorted = conv_length_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        rnn_input = pack_padded_sequence(encoder_hidden_sorted, conv_length_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=True,total_length=seq_len)
        # outputs: [batch_size, max_conversation_length, context_size]
#        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                    hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):

        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden
