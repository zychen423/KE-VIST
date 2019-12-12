''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import model.Constants as Constants
from model.Layers import EncoderLayer, DecoderLayer, ContextAttentionLayer, Attn
import torch.nn.functional as F

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

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        #self.position_enc = nn.Embedding.from_pretrained(
        #    get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #    freeze=True)

        #image order embedding
        #self.img_order_enc = nn.Embedding(
        #    6, d_word_vec, padding_idx=Constants.PAD)
        self.img_order_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(6, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_seq, src_pos, src_sen_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- src_seq build mask
        src_seq_m = src_sen_pos
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq_m, seq_q=src_seq_m)
        non_pad_mask = get_non_pad_mask(src_seq_m)

        # -- Forward
        enc_output = self.dropout(src_seq + self.img_order_enc(src_sen_pos))
        #ienc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
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

class ObjectActionEncoder(nn.Module):
    ''' Encoding object-action pair '''
    def __init__(self, vocab_size, d_model):
        super(ObjectActionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.BiLSTM = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(d_model*2, d_model)

    def forward(self, x):
        B, img_num, obj_num, seq_len = x.size()
        x = x.view(B*img_num*obj_num, seq_len)
        x = F.dropout(self.embedding(x), 0.1)
        x, _ = self.BiLSTM(x) # B*img_num*obj_num, seq_len, 2*d_model
        x = F.relu(self.fc(x)) # B*img_num*obj_num, seq_len, d_model
        x = x.transpose(2, 1)  # B*img_num*obj_num, d_model, seq_len
        x = F.max_pool1d(x, x.size(2)).squeeze() # B*img_num*obj_num, d_model

        return x.view(B, img_num, obj_num, -1) # B, img_num, obj_num, d_model

class CommonSensePathEncoder(nn.Module):
    ''' Encoding common sense path between 2 images '''
    def __init__(self, vocab_size, d_model):
        super(CommonSensePathEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.BiLSTM = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(d_model*2, d_model)

    def forward(self, x):
        B, img_num, obj_num, seq_len = x.size()
        x = x.view(B*img_num*obj_num, seq_len)
        x = F.dropout(self.embedding(x), 0.1)
        x, _ = self.BiLSTM(x) # B*img_num-1*path_num, path_len, 2*d_model
        x = F.relu(self.fc(x)) # B*img_num-1*path_num, path_len, d_model
        x = x.transpose(2, 1)  # B*img_num-1*path_num, d_model, path_len
        x = F.max_pool1d(x, x.size(2)).squeeze() # B*img_num-1*path_num, d_model

        return x.view(B, img_num, obj_num, -1) # B, img_num-1, path_num, d_model

class ImageFeatureEncoder(nn.Module):
    ''' Encoding Image Features '''
    def __init__(self, d_model):
        super(ImageFeatureEncoder, self).__init__()
        self.img2enc = nn.Linear(2048, d_model)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        B, img_num, obj_num, fc_feat_size = x.size()
        x = self.img2enc(x)
        x = self.bn(x.view(B*img_num*obj_num, -1))

        return x.view(B, img_num, obj_num , -1) # B, img_num, obj_num, d_model

class AttentionBlock(nn.Module):
    ''' BiDAF of objects in an image and all images (and Common sense attention)'''

    def __init__(self, d_model):
        super(AttentionBlock, self).__init__()
        self.bidaf = ContextAttentionLayer(d_model)
        # future work adding commen sense layer
        self.img_order_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(6, d_model, padding_idx=0),
            freeze=True)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, C, Q, C_pos, Q_pos):

        C = self.dropout(C + self.img_order_enc(C_pos))
        Q = self.dropout(Q + self.img_order_enc(Q_pos))
        C_mask = C_pos.ne(Constants.PAD)
        Q_mask = Q_pos.ne(Constants.PAD)
        x = self.bidaf(C, Q, C_mask, Q_mask) # B, d_model, img_num*obj_num
        return x.transpose(1,2)

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        #sentence positional embedding
        self.sen_position_enc = nn.Embedding(
            6, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, tgt_sen_pos, src_sen_pos, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        obj_num = src_sen_pos.size(-1)
        # -- src_seq build mask
        src_seq_m = src_sen_pos
        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq_m, seq_q=tgt_seq)
        # -- Forward
        #dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = dec_output + self.sen_position_enc(tgt_sen_pos)

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


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, max_length, vocab_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length-1
        # Define layers
        self.tgt_word_emb = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, tgt_seq, tgt_pos, tgt_sen_pos, src_sen_pos, enc_output, return_attns=False):
        # Get the embedding of the current input word (last output word)
        output = []
        B, L = tgt_seq.size()
        mask = src_sen_pos.eq(Constants.PAD)
        enc_output = enc_output.transpose(0,1)
        word_embedded = self.tgt_word_emb(tgt_seq).transpose(0,1) # (L,B,H)
        word_embedded = self.dropout(word_embedded)
        hidden = self.initHidden(B, tgt_seq.device)
        for i in range(L):
            dec_out, hidden = self._forward(word_embedded[i].view(1,B,-1), hidden, enc_output, mask)
            output.append(dec_out)

        output = torch.stack(output).contiguous()  # (L,B,H)
        return output.transpose(0,1),  # (B,L,H)

    def _forward(self, word_embedded, last_hidden, encoder_outputs, mask=None):
        '''
        :param word_input:
            word input for current time step, in shape (1*B*V)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs, mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # Return final output, hidden state
        return output, hidden

    def initHidden(self, B, device):
        """
        :param zero_hidden:
            zero hidden stat of the decoder, in shape (layers*direction*B*H)
        """
        return torch.zeros(self.n_layers*1, B, self.hidden_size, device=device)

class ImageTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, n_concept_vocab, enc_len_max_seq, dec_len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            decoder="transformer",
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            self_att=True, global_att=True, concept_path=True):

        super().__init__()

        self.object_encoder = ImageFeatureEncoder(d_model)

        self.path_encoder = CommonSensePathEncoder(n_concept_vocab, d_model)

        self.common_sense_att_block = AttentionBlock(d_model)

        self.encoder = Encoder(
            n_src_vocab=n_tgt_vocab, len_max_seq=enc_len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        if decoder=="transformer":
            self.decoder = Decoder(
                n_tgt_vocab=n_tgt_vocab, len_max_seq=dec_len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout)
        else:
            self.decoder = BahdanauAttnDecoderRNN(d_model, dec_len_max_seq, n_tgt_vocab)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        self.is_global_att=global_att
        self.is_concept_path=concept_path
        self.is_self_att=self_att

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def encode_block(self,src_seq, src_pos, src_sen_pos, cs_path, cs_path_pos):
        object_embedd = self.object_encoder(src_seq)

        B, img_num, obj_num, d_model = object_embedd.size()
        enc_embedd = object_embedd
        # intra-image self-attention
        if self.is_global_att:
            # inter-image self-attention
            enc_embedd = enc_embedd.view(B*img_num, obj_num, d_model)
            src_sen_pos = src_sen_pos.view(B*img_num, obj_num)
            enc_embedd, *_ = self.encoder(enc_embedd, src_pos, src_sen_pos)
        if self.is_self_att:
            enc_embedd = enc_embedd.view(B*img_num, obj_num, d_model)
            src_sen_pos = src_sen_pos.view(B*img_num, obj_num)
            enc_embedd, *_ = self.encoder(enc_embedd, src_pos, src_sen_pos)
        if self.is_concept_path:
            concept_embedd = self.path_encoder(cs_path)
            # adding common sense information
            enc_embedd = enc_embedd.view(B*img_num, obj_num, d_model)
            src_sen_pos = src_sen_pos.view(B*img_num, obj_num)
            concept_embedd = concept_embedd.view(B*(img_num), 10, d_model)
            cs_path_pos = cs_path_pos.view(B*(img_num), 10)
            enc_embedd = self.common_sense_att_block(enc_embedd, concept_embedd, src_sen_pos, cs_path_pos)
        enc_output = enc_embedd.contiguous().view(B*img_num, obj_num, d_model)
        src_sen_pos = src_sen_pos.view(B*img_num,obj_num)

        return enc_output, src_sen_pos

    def forward(self, src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, cs_path, cs_path_pos):

        tgt_seq, tgt_pos, tgt_sen_pos = tgt_seq[:, :-1], tgt_pos[:, :-1], tgt_sen_pos[:, :-1]

        enc_output, src_sen_pos = self.encode_block(src_seq, src_pos, src_sen_pos, cs_path, cs_path_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, tgt_sen_pos, src_sen_pos, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))
