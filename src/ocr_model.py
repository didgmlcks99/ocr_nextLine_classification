from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(
        self,
        word_vec_size=200,
        hidden_size=100,
        n_layers=3,
        dropout_p=0.2,
    ):

        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        # self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=self.word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):

        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)

        y = x[:, -1]

        return y

class OCR(nn.Module):

    def __init__(
        self,
        input_size=None,
        word_vec_size=200,
        hidden_size=100,
        n_classes=2,
        n_layers=3,
        dropout_p=0.2,
        pretrained_embedding=None,
        freeze_embedding=False,
    ):

        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.pretrained_embedding = pretrained_embedding
        self.freeze_embedding = freeze_embedding

        super(OCR, self).__init__()

        if pretrained_embedding is not None:
            print("doing with pretrained model!!!")
            self.input_size, self.word_vec_size = pretrained_embedding.shape
            self.emb = nn.Embedding.from_pretrained(pretrained_embedding,
                                                    freeze=freeze_embedding)
        else:
            print("doing without pretrained model!!!")
            self.word_vec_size = word_vec_size
            self.emb = nn.Embedding(input_size, word_vec_size)

        self.lstm1 = RNN(
            self.word_vec_size,
            self.hidden_size,
            self.n_layers,
            self.dropout_p
        )

        self.lstm2 = RNN(
            self.word_vec_size,
            self.hidden_size,
            self.n_layers,
            self.dropout_p
        )

        self.fc1 = nn.Linear(
            self.hidden_size*2*2, 
            self.n_classes
        )

        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, x1, x2):

        a = self.emb(x1).float()
        b = self.emb(x2).float()
        
        a = self.lstm1(a)
        b = self.lstm2(b)

        y = torch.cat((a, b), 1)

        y = self.fc1(self.dropout(F.relu(y)))

        return y