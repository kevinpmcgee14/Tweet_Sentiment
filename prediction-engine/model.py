import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def drop_mask(x, sz, p):
  return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        seq_len_dims = (x.size(0), 1, x.size(2))
        m = drop_mask(x, seq_len_dims, self.p)
        return x * m

class WeightDroput(nn.Module):

    def __init__(self, module, weight_p=[0.], layer_names=['weight_hh_l0'], training=True):
        super().__init__()
        self.module = module
        self.weight_p = weight_p 
        self.layer_names = layer_names
        self.training = training
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

class EmbeddingDropout(nn.Module):

    def __init__(self, embed, embed_p):
        super().__init__()
        self.emb = embed
        self.embed_p = embed_p
        self.pad_idx = self.emb.padding_idx 
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            sz = (self.emb.weight.size(0), 1)
            mask = drop_mask(self.emb.weight.data, sz, self.embed_p)
            mask_embed = self.emb.weight * mask
        else:
            mask_embed = self.emb.weight
        if scale:
            mask_embed._mul(scale)
        kwargs = {
            'padding_idx': self.pad_idx,
            'max_norm': self.emb.max_norm,
            'norm_type': self.emb.norm_type,
            'scale_grad_by_freq': self.emb.scale_grad_by_freq,
            'sparse': self.emb.sparse
        }
        return F.embedding(words, mask_embed, **kwargs)


class AWD_LSTM(nn.Module):

    def __init__(self, vocab_size, embed_size, n_hid, n_layers, pad_token, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.bs = 1
        self.emb_sz = embed_size
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.pad_token = pad_token
        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [nn.LSTM(embed_size if l == 0 else n_hid, (n_hid if l != n_layers -1 else embed_size), 1, batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDroput(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-0.1, 0.1)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        mask = (input == self.pad_token)
        lengths = sl - mask.long().sum(1)
        n_empty = (lengths == 0).sum()
        if bs > n_empty > 0:
            input = input[:-n_empty]
            lengths = lengths[:-n_empty]
            self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
    
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths.cpu(), batch_first=True, enforce_sorted=False)
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output = pad_packed_sequence(raw_output, batch_first=True, padding_value=pad_token, total_length=sl)[0]
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
            new_hidden.append(new_h)
        self.hidden = self.to_detach(new_hidden)
        return raw_outputs, outputs, mask

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

    def to_detach(self, h):
        return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)












class LinearClassifier(nn.Module):

    def __init__(self, layers, drops):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += self.bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def forward(self, input):
        raw_outputs,outputs,mask = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        nn.AvgPool1d()
        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])
        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) #Concat pooling.
        x = self.layers(x)
        return x

    def bn_drop_lin(self, n_in, n_out, bn=True, p=0., actn=None):
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: 
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers


class SentenceEncoder(nn.Module):

    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt= bptt
        self.module = module
        self.pad_idx = pad_idx

    def pad_tensor(self, t, bs):
        if t.size(0) < bs:
            return torch.cat([t, self.pad_idx + t.new_zeros(bs-t.size(0), *t.shape[1:])])
        return t

    def concat(self, arrs, bs):
        return [torch.cat([self.pad_tensor(l[si],bs) for l in arrs], dim=1) for si in range(len(arrs[0]))]
  
    def forward(self, input):
        bs,sl = input.size()
        self.module.bs = bs
        self.module.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r,o,m = self.module(input[:,i: min(i+self.bptt, sl)])
            masks.append(self.pad_tensor(m, bs))
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs, bs),self.concat(outputs, bs),torch.cat(masks,dim=1)

class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()


bptt = 182 #tok_objs.max_seq
vocab_sz = 60002 #len(tok.vocab)
emb_sz = 300
n_hid = 300
n_layers = 2
pad_token = 1 #tok.vocab.itos.index('<pad>')
n_out = 2
output_p = 0.4
c_layers = [50]
c_drops = [0.1] * len(c_layers)
c_layers = [3 * emb_sz] + c_layers + [n_out] 
c_drops = [output_p] + c_drops

rnn_encoder = AWD_LSTM(vocab_sz, emb_sz, n_hid, n_layers, pad_token)
enc = SentenceEncoder(rnn_encoder, bptt, pad_idx=pad_token)
classifier = LinearClassifier(c_layers, c_drops)

model = SequentialRNN(enc, classifier)