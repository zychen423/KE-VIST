import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 ninp,
                 nhid,
                 dropout=0.5,
                 tie_weights=False,
                 pad_token=None):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn1 = nn.LSTM(ninp, nhid, 1,
                            dropout=dropout,
                            batch_first=True)
        self.rnn2 = nn.LSTM(ninp, nhid, 1,
                            dropout=dropout,
                            batch_first=True)
        self.rnn3 = nn.LSTM(ninp, nhid, 1,
                            dropout=dropout,
                            batch_first=True)
        self.rnn4 = nn.LSTM(ninp, nhid, 1,
                            dropout=dropout,
                            batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.PAD = pad_token

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, lengths, hidden=None):
        emb = self.drop(self.encoder(input))
        emb = nn.utils.rnn.pack_padded_sequence(emb,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=False)
        output, hidden = self.rnn1(emb, hidden)
        emb, lengths = nn.utils.rnn.pad_packed_sequence(
            emb, batch_first=True, padding_value=self.PAD)
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, padding_value=self.PAD)
        output_res = emb + output
        output_res = nn.utils.rnn.pack_padded_sequence(output_res,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)

        output, hidden = self.rnn2(output_res, None)
        output_res, lengths = nn.utils.rnn.pad_packed_sequence(
            output_res, batch_first=True, padding_value=self.PAD)
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, padding_value=self.PAD)
        output_res = output_res + output
        output_res = nn.utils.rnn.pack_padded_sequence(output_res,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)

        output, hidden = self.rnn3(output_res, None)
        output_res, lengths = nn.utils.rnn.pad_packed_sequence(
            output_res, batch_first=True, padding_value=self.PAD)
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, padding_value=self.PAD)
        output_res = output_res + output
        output_res = nn.utils.rnn.pack_padded_sequence(output_res,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)

        output, hidden = self.rnn4(output_res, None)
        output_res, lengths = nn.utils.rnn.pad_packed_sequence(
            output_res, batch_first=True, padding_value=self.PAD)
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, padding_value=self.PAD)
        output = output + output_res

        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden, lengths

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
