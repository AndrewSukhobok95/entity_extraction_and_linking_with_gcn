import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List



class NYTjsonDataset(Dataset):
    def __init__(self, data, bertinizer, eraligner):
        self.data = data
        self.bertinizer = bertinizer
        self.eraligner = eraligner

    def __getitem__(self, index):
        observation = self.data[index]

        sentence = observation["sentText"]
        entityMentions = observation["entityMentions"]
        relationMentions = observation["relationMentions"]

        tokens_wp, tokens_ids_wp_tensor, segments_ids_wp_tensors, tokens_base = self.bertinizer.tokenize(sentence)
        ne_tensor, rel_tensor = self.eraligner.get_ne_rel_tensors(tokens_base, entityMentions, relationMentions)

        bert_embeddings = self.bertinizer.get_embeddings(tokens_ids_wp_tensor, segments_ids_wp_tensors)
        bert_avg_embeddings = self.bertinizer.average_wp_embeddings(bert_embeddings, tokens_wp)

        return bert_avg_embeddings, ne_tensor, rel_tensor

    def __len__(self):
        return len(self.data)


def pad_rel_tensors(rel_tensors: List[torch.tensor],
                    batch_first: bool = False,
                    padding_value: int = -1) -> torch.tensor:
    '''
    Pads every relation matrix to the one with max seq_len
    :param rel_tensors: List of relation matrices
                        Each matrix has size -> ( seq_len_i, seq_len_i )
    :param batch_first: Flag to put batch first
    :param padding_value: Value to pad the matrix with
    :return: tensor with padded matrices:
             - If batch_first = True: tensor.size() -> ( batch, max_seq_len, max_seq_len )
             - If batch_first = False: tensor.size() -> ( max_seq_len, max_seq_len, batch )
    '''
    n_obs = len(rel_tensors)
    max_seq_len = rel_tensors[0].size(0)

    if batch_first:
        padded_rel_tensors = torch.ones(n_obs, max_seq_len, max_seq_len) * padding_value
        padded_rel_tensors[0,:,:] = rel_tensors[0]
    else:
        padded_rel_tensors = torch.ones(max_seq_len, max_seq_len, n_obs) * padding_value
        padded_rel_tensors[:,:,0] = rel_tensors[0]

    for i in range(1, n_obs-1):
        rt = rel_tensors[i]
        if batch_first:
            padded_rel_tensors[i, :rt.size(0), :rt.size(1)] = rt
        else:
            padded_rel_tensors[:rt.size(0), :rt.size(1), i] = rt

    return padded_rel_tensors

def collate_fn(batch):
    '''
    Convert input batch to the input of the network
    :param batch: Consists of n (defined by DataLoader) observations
                  Each observation is expected to be output from NYTjsonDataset:
                  - bert_avg_embeddings: tensor.size() -> (seq_len, embedding_size)
                  - ne_tensor: tensor.size() -> (seq_len)
                  - rel_tensor: tensor.size() -> (seq_len, seq_len)
    :return: Prepared data tensors:
             - b_avgemb_tensor: size() -> ( seq_len, batch, embedding_size )
             - b_ne_tensor: size() -> ( seq_len, batch )
             - b_rel_tensor: size is regulated by batch_first parameter of pad_rel_tensors
                - If batch_first = True: tensor.size() -> ( batch, seq_len, seq_len )
                - If batch_first = False: tensor.size() -> ( seq_len, seq_len, batch )
    '''
    n_obs = len(batch)
    sentences = []
    ne_output = []
    rel_output = []
    sentences_length = []
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    for i in range(n_obs):
        sentences.append(batch[i][0])
        ne_output.append(batch[i][1])
        rel_output.append(batch[i][2])
        sentences_length.append(batch[i][0].size(0))
    b_avgemb_tensor = nn.utils.rnn.pad_sequence(sentences, batch_first=False, padding_value=0)
    b_ne_tensor = nn.utils.rnn.pad_sequence(ne_output, batch_first=False, padding_value=-1)
    b_rel_tensor = pad_rel_tensors(rel_output, batch_first=False, padding_value=-1)
    return b_avgemb_tensor, b_ne_tensor, b_rel_tensor, sentences_length


if __name__=="__main__":

    from data_processing.BERTinizer import SentenceBERTinizer
    from data_processing.data_prep import EntityRelationsAligner, get_dataset

    sentbertnizer = SentenceBERTinizer()

    data_nyt_train, NE_LIST, REL_LIST = get_dataset("./../data/preproc_NYT_json/train.json", sentbertnizer.tokenizer)

    er_aligner = EntityRelationsAligner(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)

    trainset = NYTjsonDataset(data_nyt_train, sentbertnizer, er_aligner)

    trainloader = DataLoader(trainset, batch_size=2, collate_fn=collate_fn)

    t = next(iter(trainloader))

    print(t[0].size())
    print(t[1].size())
    print(t[2].size())

    print("+ done!")
