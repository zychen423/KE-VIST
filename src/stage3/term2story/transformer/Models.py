''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
#from Sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    #print('torch.FloatTensor(sinusoid_table)', torch.FloatTensor(sinusoid_table).shape)
    return torch.FloatTensor(sinusoid_table)

#LDPE length-position/10000
def get_sinusoid_encoding_LDPE_table(n_position, d_hid, padding_idx=None, length = 6):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return (length- position) / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    #print('torch.FloatTensor(sinusoid_table)', torch.FloatTensor(sinusoid_table).shape)
    return torch.FloatTensor(sinusoid_table)

#LRPE position/lenth
def get_sinusoid_encoding_LRPE_table(n_position, d_hid, padding_idx=None, length = 51):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return (position/length) / np.power(100, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    #print('torch.FloatTensor(sinusoid_table)', torch.FloatTensor(sinusoid_table).shape)
    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_percentage_encoding(seq):
    percentage = ((seq[0][0]-1)/5)+1
    for s in seq:
        s[s > 0] = percentage
    return seq

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, positional = 'Default'):

        super().__init__()
        print('Encoder...positional',positional)
        self.len_max_seq = len_max_seq
        self.d_word_vec = d_word_vec
        self.d_k = d_k
        
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        
        
        #self.position_enc = nn.Embedding.from_pretrained(
        #    get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #    freeze=True)
        
        if positional == 'LRPE':
            self.position_enc_LC = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_LRPE_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)
        elif positional == 'LDPE':
            self.position_enc_LC = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_LDPE_table(n_position, d_word_vec, padding_idx=0),
                freeze=True)
        else:
            self.position_enc_LC = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
                freeze=True)

        #LDPE and LRPE
        

        

        
        #sentence positional embedding
        self.sen_position_enc = nn.Embedding(
            6, d_word_vec, padding_idx=Constants.PAD)
        #self.sen_position_enc = nn.Embedding.from_pretrained(
        #    get_sinusoid_encoding_table(10, d_word_vec, padding_idx=0),
        #    freeze=True)
        
        
        #Positional Encoding to Control Output Sequence Length: length-ratio positional encoding (lrpe)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        

    def forward(self, src_seq, src_pos, src_sen_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward        
        enc_output = self.src_word_emb(src_seq) + self.position_enc_LC(src_sen_pos)
        
        #enc_output = torch.cat((self.src_word_emb(src_seq),self.get_percentage_encoding(src_sen_pos)))       
        #enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        #enc_output = enc_output + self.sen_position_enc(src_sen_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, positional = 'Default'):
        
        super().__init__() 
        
        print('Decoder...positional',positional)
        self.len_max_seq = len_max_seq
        self.d_word_vec = d_word_vec
        self.d_k = d_k

        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
                freeze=True) 
        
        #LDPE and LRPE
        if positional == 'LRPE':
            self.position_enc_LC = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_LRPE_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)
        elif positional == 'LDPE':
            self.position_enc_LC = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_LDPE_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)
        else:
            self.position_enc_LC = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
                freeze=True)
      
        
        #sentence positional embedding
        self.sen_position_enc = nn.Embedding(
            6, d_word_vec, padding_idx=Constants.PAD)
                      
       
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    
    def forward(self, tgt_seq, tgt_pos, tgt_sen_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
              
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = dec_output + self.position_enc_LC(tgt_sen_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, encode_len_max_seq, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True, 
            positional = 'Default'):
        super().__init__()
        
        print('In Transformer ... positional:', positional)
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=encode_len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, positional = positional)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, positional = positional)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos):

        tgt_seq, tgt_pos, tgt_sen_pos = tgt_seq[:, :-1], tgt_pos[:, :-1], tgt_sen_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos, src_sen_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, tgt_sen_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
