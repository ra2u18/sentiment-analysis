import sys
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from torch import nn
from utils_bert import pad_sents
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded

def sents_to_tensor(tokenizer, sents, device):
    """
    :param tokenizer: BertTokenizer
    :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences), reversely sorted
    :param device: torch.device
    :return: sents_tensor: torch.Tensor, shape(batch_size, max_sent_length), reversely sorted
    :return: masks_tensor: torch.Tensor, shape(batch_size, max_sent_length), reversely sorted
    :return: sents_lengths: torch.Tensor, shape(batch_size), reversely sorted
    """
    tokens_list = [tokenizer.tokenize(sent) for sent in sents]
    sents_lengths = [len(tokens) for tokens in tokens_list]
    
    tokens_list_padded = pad_sents(tokens_list, '[PAD]')
    sents_lengths = torch.tensor(sents_lengths, device='cpu')

    masks = []
    for tokens in tokens_list_padded:
        mask = [0 if token=='[PAD]' else 1 for token in tokens]
        masks.append(mask)
    masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
    sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

    return sents_tensor, masks_tensor, sents_lengths

class CustomBertLSTMModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, lstm_hidden_size=None):
        """
        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param lstm_hidden_size: int
        """

        super(CustomBertLSTMModel, self).__init__()

        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, sents):
        """
        :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(encoded_layers, sents_lengths))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertLSTMModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, lstm_hidden_size=self.lstm_hidden_size,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

class CustomBertLSTMAttentionModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, lstm_hidden_size=None):
        """
        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param lstm_hidden_size: int
        """

        super(CustomBertLSTMAttentionModel, self).__init__()

        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)

        if not lstm_hidden_size:
            self.lstm_hidden_size = self.bert.config.hidden_size
        else:
            self.lstm_hidden_size = lstm_hidden_size
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, sents):
        """
        :param sents: list[str], list of sentences (NOTE: untokenized, continuous sentences)
        :return: pre_softmax, torch.tensor of shape (batch_size, n_class)
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=False)
        encoded_layers = encoded_layers.permute(1, 0, 2)

        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(encoded_layers, sents_lengths))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertLSTMModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, lstm_hidden_size=self.lstm_hidden_size,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

class CustomBertConvModel(nn.Module):

    def __init__(self, bert_config, device, dropout_rate, n_class, out_channel=16):
        """
        :param bert_config: str, BERT configuration description
        :param device: torch.device
        :param dropout_rate: float
        :param n_class: int
        :param out_channel: int, NOTE: out_channel per layer of BERT
        """

        super(CustomBertConvModel, self).__init__()

        self.bert_config = bert_config
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.out_channel = out_channel
        self.bert = BertModel.from_pretrained(self.bert_config)
        self.out_channels = self.bert.config.num_hidden_layers*self.out_channel
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)
        self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers,
                              out_channels=self.out_channels,
                              kernel_size=(3, self.bert.config.hidden_size),
                              groups=self.bert.config.num_hidden_layers)
        self.hidden_to_softmax = nn.Linear(self.out_channels, self.n_class, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.device = device

    def forward(self, sents):
        """
        :param sents:
        :return:
        """

        sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
        encoded_layers, pooled_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor,
                                                  output_all_encoded_layers=True)

        encoded_stack_layer = torch.stack(encoded_layers, 1)  # (batch_size, channel, max_sent_length, hidden_size)
        conv_out = self.conv(encoded_stack_layer)  # (batch_size, channel_out, some_length, 1)
        conv_out = torch.squeeze(conv_out, dim=3)  # (batch_size, channel_out, some_length)
        conv_out, _ = torch.max(conv_out, dim=2)  # (batch_size, channel_out)
        pre_softmax = self.hidden_to_softmax(conv_out)

        return pre_softmax

    @staticmethod
    def load(model_path: str, device):
        """ Load the model from a file.
        @param model_path (str): path to model
        @return model (nn.Module): model with saved parameters
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CustomBertConvModel(device=device, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(bert_config=self.bert_config, out_channel=self.out_channel,
                         dropout_rate=self.dropout_rate, n_class=self.n_class),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)