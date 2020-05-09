import torch
from torch.utils.data import Dataset, DataLoader


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

def collate_fn():
    return