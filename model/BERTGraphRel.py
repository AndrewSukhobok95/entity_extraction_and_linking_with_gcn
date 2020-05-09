import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, hidden_size=256):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 2)

    def forward(self, x, adj, is_relu=True):
        out = self.fc(x)

        # Make permutations for matrix multiplication
        # Assuming batch_first = False
        out = out.permute(1, 0, 2) # to: batch, seq_len, hidden
        adj = adj.permute(2, 0, 1) # to: batch, seq_len, seq_len

        out = torch.bmm(adj, out).permute(1, 0, 2) # to: seq_len, batch, hidden

        if is_relu == True:
            out = F.relu(out)

        return out


class BERTGraphRel(nn.Module):
    def __init__(self, num_ne, num_rel, embedding_size, hidden_size=256, n_rnn_layers=2, dropout_prob=0.1):
        super(BERTGraphRel, self).__init__()

        self.num_ne = num_ne
        self.num_rel = num_rel
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.dropout_prob = dropout_prob

        self.batch_first = False

        # Encoding part
        self.rnn_enc = nn.GRU(self.embedding_size,
                              self.hidden_size,
                              bidirectional=True,
                              num_layers=self.n_rnn_layers,
                              batch_first=self.batch_first,
                              dropout=self.dropout_prob)
        self.fc_enc = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        # Prediction part
        # Prediction NE
        self.rnn_ne = nn.GRU(self.hidden_size * 2,
                             self.hidden_size,
                             bidirectional=True,
                             batch_first=self.batch_first)
        self.fc_ne = nn.Linear(self.hidden_size * 2, self.num_ne)
        # Prediction REL
        self.fc_trel_0 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_trel_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_rel = nn.Linear(self.hidden_size * 2, self.num_rel)

        # GCN part
        self.gcn_fw = nn.ModuleList([GCN(self.hidden_size * 2) for _ in range(self.num_rel)])
        self.gcn_bw = nn.ModuleList([GCN(self.hidden_size * 2) for _ in range(self.num_rel)])

    def prediction_block(self, x):
        # NE part
        x_ne, h_ne = self.rnn_ne(x)
        out_ne = self.fc_ne(x_ne)

        # REL part
        trel_0 = F.relu(self.fc_trel_0(x))
        trel_1 = F.relu(self.fc_trel_1(x))

        trel_0 = trel_0.view((trel_0.shape[0], 1, trel_0.shape[1], trel_0.shape[2]))
        trel_0 = trel_0.expand((trel_0.shape[0], trel_0.shape[0], trel_0.shape[2], trel_0.shape[3]))
        trel_1 = trel_1.view((1, trel_1.shape[0], trel_1.shape[1], trel_1.shape[2]))
        trel_1 = trel_1.expand((trel_1.shape[1], trel_1.shape[1], trel_1.shape[2], trel_1.shape[3]))
        trel = torch.cat([trel_0, trel_1], dim=3)

        out_rel = self.fc_rel(trel)

        return out_ne, out_rel

    def forward(self, x):
        bf_dim = int(self.batch_first)

        x, h = self.rnn_enc(x)
        out_enc = self.fc_enc(x)

        out_ne_p1, out_rel_p1 = self.prediction_block(out_enc)

        adj_fw = nn.functional.softmax(out_rel_p1, dim=3)
        adj_bw = adj_fw.transpose(0 + bf_dim, 1 + bf_dim)

        gcn_outs = []
        for i in range(self.num_rel):
            out_fw = self.gcn_fw[i](out_enc, adj_fw[:, :, :, i])
            out_bw = self.gcn_bw[i](out_enc, adj_bw[:, :, :, i])
            gcn_outs.append(torch.cat([out_fw, out_bw], dim=2))

        out_enc_p2 = out_enc
        for i in range(self.num_rel):
            out_enc_p2 += gcn_outs[i]

        out_ne_p2, out_rel_p2 = self.prediction_block(out_enc_p2)

        return out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2




if __name__=="__main__":

    num_ne = 7
    num_rel = 8
    embedding_size = 5
    hidden_size = 6

    sentence_1 = torch.ones((7, embedding_size)) * 2
    sentence_2 = torch.ones((10, embedding_size)) * 3
    sequences = [sentence_1, sentence_2]
    batch = nn.utils.rnn.pad_sequence(sequences,
                                      batch_first=False,
                                      padding_value=0)

    print("===================================================")
    print("+ Input size:", batch.size())

    model = BERTGraphRel(num_ne=num_ne,
                         num_rel=num_rel,
                         embedding_size=embedding_size,
                         hidden_size=hidden_size)

    out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2 = model(batch)

    print("+ Output NE p1 size:", out_ne_p1.size())
    print("+ Output REL p1 size:", out_rel_p1.size())
    print("+ Output NE p2 size:", out_ne_p2.size())
    print("+ Output REL p2 size:", out_rel_p2.size())
    print("===================================================")

    print("+ done!")

