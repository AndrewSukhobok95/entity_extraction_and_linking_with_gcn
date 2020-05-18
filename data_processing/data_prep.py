import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
import json
from typing import List, Tuple
import tqdm

class tgtEntRelConstructor(object):
    def __init__(self, tokenizer, ne_tags: List[str], rel_tags: List[str]):
        # tokenizer must have base_tokenize method similar to SentenceBERTnizer
        self.tokenizer = tokenizer
        # NE
        self.NE_tags = ne_tags
        self.NE_biotags = self._get_bio_tags()
        self.NE_vsize = len(self.NE_biotags)
        self.NE_bio_to_index_dict = self._get_bio_to_index()
        # REL
        self.REL_tags = rel_tags
        self.REL_mod_tags = self._get_mod_rel_tags()
        self.REL_vsize = len(self.REL_mod_tags)
        self.REL_mod_to_index_dict = self._get_mod_rel_to_index()

    def _get_bio_tags(self) -> List[str]:
        '''
        Transfer entity labels to BIO scheme
        :return: [ "O", "B-...", "I-...", "B-...", ... ]
        '''
        NE_biotags = ["O"]
        for ne in self.NE_tags:
            NE_biotags.append("B-" + ne)
            NE_biotags.append("I-" + ne)
        return NE_biotags

    def _get_bio_to_index(self) -> dict:
        '''
        Create dict with indexes of vector for NE
        Which value of vector stands for the particular BIO-NE
        :return: { "O": 0, "B-...": 1, "I-...": 2, ... }
        '''
        d = {}
        for i in range(self.NE_vsize):
            d[self.NE_biotags[i]] = i
        return d

    def _get_mod_rel_tags(self) -> List[str]:
        '''
        Add relation "O" standing for "no relation"
        :return: [ "O", "rel_1", "rel_2", ... ]
        '''
        REL_mod_tags = ["O"]
        for r in self.REL_tags:
            REL_mod_tags.append(r)
        return REL_mod_tags

    def _get_mod_rel_to_index(self) -> dict:
        '''
        Create dict with indexes of vector for relations
        Which value of vector stands for the particular relation
        :return: { "O": 0, "rel_1": 1, "rel_2": 2, ... }
        :return:
        '''
        d = {}
        for i in range(self.REL_vsize):
            d[self.REL_mod_tags[i]] = i
        return d

    def find_entities_indexes(self, tokens: List[str], entities: List[dict]) -> List[dict]:
        '''
        Find indices of tokens, which are NE
        :param tokens: [ "token_0", "token_1", "token_2", ... ]
        :param entities: [ { "text": "textA textB textC",
                             "label": "NE_j" }, ... ]
        :return: modified entities: [ { "text": "textA textB textC",
                                        "label": "NE_j",
                                        "tokens": [ "textA", "textB", "textC" ],
                                        "n_tokens": 3,
                                        "tokens_index": [i, i+1, i+2] }, ... ]
        '''
        for e in entities:
            e["tokens"] = self.tokenizer.base_tokenize(e["text"])
            e["n_tokens"] = len(e["tokens"])
        for i in range(len(tokens)):
            for e in entities:
                e_tokens = np.array(e["tokens"])
                s_tokens = np.array(tokens[i:(i + e["n_tokens"])])
                if (e_tokens.size == s_tokens.size) and np.all(e_tokens == s_tokens):
                    e["tokens_index"] = list(range(i, (i + e["n_tokens"])))
        return entities

    def get_ne_tensor(self, n_tokens: int, entities: List[dict]) -> torch.tensor:
        '''
        Create a torch.tensor for the indices of entity tokens in the sentence
        :param n_tokens: number of tokens in sentence
        :param entities: result of self.ind_entities_indexes
                         [ { "text": "textA textB textC",
                             "label": "NE_j",
                             "tokens": [ "textA", "textB", "textC" ],
                             "n_tokens": 3,
                             "tokens_index": [i, i+1, i+2] }, ... ]
        :return: tensor.size() -> ( n_tokens )
        '''
        ne_array = np.ones(n_tokens) * self.NE_bio_to_index_dict["O"]
        for e in entities:
            tind = e["tokens_index"]
            tb_index = self.NE_bio_to_index_dict["B-" + e["label"]]
            ti_index = self.NE_bio_to_index_dict["I-" + e["label"]]

            ne_array[tind[0]] = tb_index

            if e["n_tokens"] > 1:
                for i in range(1, e["n_tokens"] - 1):
                    ne_array[tind[i]] = ti_index
        return torch.tensor(ne_array)

    def prep_entities_dict(self, entities: List[dict]) -> dict:
        '''
        Create dict from NE info list
        :param entities: entity list info
                [ { "text": "textA textB textC",
                    "label": "NE_j" }, ... ]
            or result of self.ind_entities_indexes
                [ { "text": "textA textB textC",
                    "label": "NE_j",
                    "tokens": [ "textA", "textB", "textC" ],
                    "n_tokens": 3,
                    "tokens_index": [i, i+1, i+2] }, ... ]
        :return: dict with entities and "text" fields as keys
                {
                    "textA textB textC": {...},
                    ...
                }
        '''
        entities_dict = {}
        for e in entities:
            entities_dict[e["text"]] = e
        return entities_dict

    def get_rel_tensor(self, n_tokens: int,
                       entities_dict: dict,
                       relations: List[dict],
                       rel_3Dtensor_out: bool=False) -> torch.tensor:
        '''
        Create torch.tensor for relations in the sentence
        :param n_tokens: number of tokens in sentence
        :param entities_dict: result of self.ind_entities_indexes
                [ { "text": "textA textB textC",
                    "label": "NE_j",
                    "tokens": [ "textA", "textB", "textC" ],
                    "n_tokens": 3,
                    "tokens_index": [i, i+1, i+2] }, ... ]
        :param relations: [{"em1Text": "Entity 1",
                            "em2Text": "Entity 2",
                            "label": "relation" }, ...]
        :param rel_3Dtensor_out: define the format of output relations tensor
        :return: tensor, defining relations between entities
            - if rel_3Dtensor_out=True: tensor.size() -> ( number of relations, n_tokens, n_tokens )
            - if rel_3Dtensor_out=False: tensor.size() -> ( n_tokens, n_tokens )
        '''
        if rel_3Dtensor_out:
            rel_array = np.ones((self.REL_vsize, n_tokens, n_tokens))
        else:
            rel_array = np.ones((n_tokens, n_tokens))
        rel_array = rel_array * self.REL_mod_to_index_dict["O"]

        for r in relations:
            e_src = r["em1Text"]
            e_tgt = r["em2Text"]
            rel = r["label"]

            rel_index = self.REL_mod_to_index_dict[rel]
            e_src_index = entities_dict[e_src]["tokens_index"][0]
            e_tgt_index = entities_dict[e_tgt]["tokens_index"][0]

            if rel_3Dtensor_out:
                rel_array[rel_index, e_src_index, e_tgt_index] = 1
            else:
                rel_array[e_src_index, e_tgt_index] = rel_index

        return torch.tensor(rel_array)

    def get_ne_rel_tensors(self, tokens: List[str],
                           entities: List[dict],
                           relations: List[dict]) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Use all the functions of the class to provide:
            - torch.tensor for indices of entity tokens in the sentence
            - torch.tensor for relations in the sentence
        :param tokens: [ "token_0", "token_1", "token_2", ... ]
        :param entities: [ { "text": "textA textB textC",
                             "label": "NE_j" }, ... ]
        :param relations: [ { "em1Text": "Entity 1",
                              "em2Text": "Entity 2",
                              "label": "relation" }, ...]
        :return: 2 tensors
            - For entities: tensor.size() -> ( n_tokens )
            - For relations:
                - Default: tensor.size() -> ( n_tokens, n_tokens )
                - Using rel_3Dtensor_out parameter of self.get_rel_tensor, could be changed to:
                    tensor.size() -> ( number of relations, n_tokens, n_tokens )
        '''
        n_tokens = len(tokens)

        entities = self.find_entities_indexes(tokens, entities)
        ne_tensor = self.get_ne_tensor(n_tokens, entities)

        entities_dict = self.prep_entities_dict(entities)
        rel_tensor = self.get_rel_tensor(n_tokens, entities_dict, relations)

        return ne_tensor, rel_tensor



def get_dataset(path, bert_wp_tokenizer):
    '''
    Temporary function for particular dataset provided by https://github.com/INK-USC/USC-DS-RelationExtraction.
    It excludes observations that:
        - Have mismatches between entities and original text
        - Are longer then 512 tokens after WordPiece tokenezation (BERT restriction)
    Also collects all mentioned entities and relation types
    :param path: path to json file
    :param bert_wp_tokenizer: BERT tokenizer
    :return:
        - Filtered data -> list of dicts
        - List of unique entity types -> list of strs
        - List of unique relation types -> list of strs
    '''
    data = []
    ne_set = set()
    rel_set = set()

    n_obs = 0
    n_broken = 0

    with open(path) as file:
        for f in tqdm.tqdm(file):
            skip_obs = False
            obs = json.loads(f)

            sentText = obs["sentText"]
            ne_mentiones = obs['entityMentions']
            rel_mentiones = obs['relationMentions']

            #marked_sentText = "[CLS] " + sentText + " [SEP]"
            #sentText_wp_tokens = bert_wp_tokenizer.tokenize(marked_sentText)
            marked_sentText = "[CLS] " + sentText.lower() + " [SEP]"
            sentText_wp_tokens = bert_wp_tokenizer.wordpiece_tokenizer.tokenize(marked_sentText)

            if len(sentText_wp_tokens) > 512:
                skip_obs = True

            for ne in ne_mentiones:
                ne_set.add(ne["label"])
                if ne["text"] not in sentText:
                    n_broken += 1
                    skip_obs = True

            for rel in rel_mentiones:
                rel_set.add(rel["label"])
                if rel["em1Text"] not in sentText:
                    n_broken += 1
                    skip_obs = True
                if rel["em2Text"] not in sentText:
                    n_broken += 1
                    skip_obs = True

            if skip_obs:
                continue

            n_obs += 1
            data.append(obs)

    print("++ Reading", path)
    print("++++ Number of added observations:", n_obs)
    print("++++ Number of broken (excluded) observations:", n_broken)

    return data, list(ne_set), list(rel_set)



if __name__=="__main__":

    _bert_wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # wiki_json_train = "./data/preproc_WikiKBP_json/train.json"
    # pubmed_json_train = "./data/preproc_PubMed_json/train.json"
    nyt_json_train = "./../data/preproc_NYT_json/train.json"
    nyt_json_test = "./../data/preproc_NYT_json/test.json"

    data_nyt_train, NE_LIST, REL_LIST = get_dataset(nyt_json_train, _bert_wp_tokenizer)
    data_nyt_test, _, _ = get_dataset(nyt_json_test, _bert_wp_tokenizer)

    obs = data_nyt_train[5]
    sentence = obs["sentText"]
    entityMentions = obs["entityMentions"]
    relationMentions = obs["relationMentions"]

    from data_processing.BERTinizer import SentenceBERTinizer
    sentbertnizer = SentenceBERTinizer()

    er_aligner = tgtEntRelConstructor(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)

    tokens_base = sentbertnizer.base_tokenize(sentence, clean_marking=False)
    ne_tensor, rel_tensor = er_aligner.get_ne_rel_tensors(tokens_base, entityMentions, relationMentions)

    print("+ Original sentence: \t", sentence)
    print("+ Tokenized sentence (without WordPiece): \t", tokens_base)
    print("+ Sentence of entities indices: \t", ne_tensor)
    print()
    print("+ Original NE: \t", entityMentions)
    print("+ BIO NE prepared: \t", er_aligner.NE_biotags)
    print()
    print("+ Size of NE tensor: \t", ne_tensor.size())
    print("+ Size of relation tensor: \t", rel_tensor.size())

    print("+ done!")

