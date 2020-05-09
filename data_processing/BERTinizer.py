import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
from typing import List, Tuple



class SentenceBERTinizer(object):
    def __init__(self, model_type="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_type)
        self.bert_model = BertModel.from_pretrained(model_type)
        self.bert_model.eval()
        self.num_hidden_layers = self.bert_model.config.num_hidden_layers
        self.embedding_size = self.bert_model.config.hidden_size

    def set_bert_to_eval_mode(self):
        self.bert_model.eval()


    def set_bert_to_train_mode(self):
        self.bert_model.train()


    def base_tokenize(self, sentence: str,
                      add_marking: bool = True,
                      clean_marking: bool = True) -> List[str]:
        '''
        Perform tokenization without splitting unknown words
        :param sentence: str sentence to tokenize
        :param add_marking: flag - if True, then ddd BERT special tokens [CLS] and [SEP]
        :param clean_marking: flag - if True, then remove [CLS] and [SEP] from the output
        :return: tokens
        '''
        if add_marking:
            sentence = "[CLS] " + sentence + " [SEP]"
        tokens = self.tokenizer.basic_tokenizer.tokenize(sentence)
        if clean_marking:
            tokens = tokens[1:len(tokens) - 1]
        return tokens


    def tokenize(self, sentence: str) -> Tuple[List[str], torch.tensor, torch.tensor, List[str]]:
        '''
        Perform tokenization with splitting unknown words with WordPiece from BERT
        :param sentence: str sentence to split
        :return:
            - tokens_wp: tokens with WordPiece splitting
            - tokens_ids_wp_tensor: tensor with indices of NE
            - segments_ids_wp_tensors: tensor with segmrnts (tensor of 1 in this case)
            - tokens_base: tokens without WordPiece splitting and removed [CLS] and [SEP]
        '''
        marked_s = "[CLS] " + sentence + " [SEP]"

        tokens_wp = self.tokenizer.tokenize(marked_s)
        tokens_ids_wp = self.tokenizer.convert_tokens_to_ids(tokens_wp)
        segments_ids_wp = [1] * len(tokens_wp)

        tokens_ids_wp_tensor = torch.tensor([tokens_ids_wp])
        segments_ids_wp_tensors = torch.tensor([segments_ids_wp])

        tokens_base = self.base_tokenize(marked_s, add_marking=False, clean_marking=False)

        return tokens_wp, tokens_ids_wp_tensor, segments_ids_wp_tensors, tokens_base

    def average_wp_embeddings(self, emb: torch.tensor,
                              wp_tokens: List[str]) -> torch.tensor:
        mask = list(map(lambda x: 1 if x[0:2]=="##" else 0, wp_tokens))
        out_dim = len(list(filter(lambda x: x==0, mask)))
        avg_embedding = torch.zeros(out_dim, emb.size(1))
        index = -1
        n = 1
        for i in range(emb.size(0)):
            if (i>0) & (mask[i]==0) & (mask[i-1]==1):
                avg_embedding[index, :] = avg_embedding[index, :] / n
                n = 1
            if mask[i]==0:
                index += 1
            if mask[i]==1:
                n += 1
            avg_embedding[index, :] += emb[i, :]
        return avg_embedding


    def get_embeddings(self, tokens_ids_tensor: torch.tensor,
                       segments_ids_tensors: torch.tensor) -> torch.tensor:

        with torch.no_grad():
            encoded_layers, encoded_output = self.bert_model(tokens_ids_tensor, segments_ids_tensors)

        # tokens_embeddings.size() -> ( layers , batch , seq_len , embedding_size )
        # tokens_embeddings = torch.stack(encoded_layers, dim=0)
        # tokens_embeddings = tokens_embeddings.permute(1,0,2)

        # tokens_embeddings.size() -> ( batch , seq_len , embedding_size )
        tokens_embeddings = encoded_layers[self.num_hidden_layers - 1].squeeze()

        return tokens_embeddings




if __name__=="__main__":

    sentbertnizer = SentenceBERTinizer()

    sentence = "I want to take some qusiakkom."

    tokens_wp, tokens_ids_wp_tensor, segments_ids_wp_tensors, tokens_base = sentbertnizer.tokenize(sentence)
    embeddings = sentbertnizer.get_embeddings(tokens_ids_wp_tensor, segments_ids_wp_tensors)
    avg_embeddings = sentbertnizer.average_wp_embeddings(embeddings, tokens_wp)

    t = ["a", "b", "##b", "c", "d", "e", "##e", "##e", "f"]
    e = torch.tensor([
        [1,2], # a    [1,2]
        [2,4], # b    [4,6]
        [6,8], # ##b
        [1,2], # c    [1,2]
        [3,4], # d    [3,4]
        [3,9], # e    [3,5]
        [1,3], # ##e
        [5,3], # ##e
        [9,9]  # f    [9,9]
    ])
    ta = sentbertnizer.average_wp_embeddings(e, t)

    print("+ Original sentence: \t", sentence)
    print("+ Tokenized sentence:\t", tokens_base)

    print("+ done!")

