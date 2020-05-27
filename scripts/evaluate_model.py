import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel

from data_processing.BERTinizer import SentenceBERTinizer
from data_processing.data_prep import tgtEntRelConstructor, get_dataset
from data_processing.data_load import jsonDataset, collate_fn
from data_processing.EntityRelationInfoCollector import InfoCollector
from model.BERTGraphRel import BERTGraphRel
from model.training import train_BERTGraphRel_model
from model.evaluation import eval_BERTGraphRel_model

# text_info_dict_path = "./../json_dicts/dumb_info.json"
# data_json_train_path = "./../data/dumb_2/dumb_train.json"
# data_json_test_path = "./../data/dumb_2/dumb_test.json"
# model_path = "./../trained_models/dumb_bertlgl_v0.pth"

text_info_dict_path = "./../json_dicts/NYT_info.json"
data_json_train_path = "./../data/preproc_NYT_json/train.json"
data_json_test_path = "./../data/preproc_NYT_json/test.json"
model_path = "./../trained_models/nyt_bertgl_v0.pth"

if __name__=="__main__":

    # BERT Prep

    print("+ Preparing BERT model.")
    sentbertnizer = SentenceBERTinizer()

    # Data Prep

    print("+ Reading data.")
    data_train, _, _ = get_dataset(data_json_train_path)
    data_test, _, _ = get_dataset(data_json_test_path)

    print("+ Preparing data.")
    info_collector = InfoCollector()

    print("+ Loading text info.")
    info_collector.load_info_dict(path=text_info_dict_path)
    num_ne = info_collector.info_dict["entity_vsize"]
    num_rel = info_collector.info_dict["rel_vsize"]
    NE_LIST = info_collector.info_dict["original_entities"]
    REL_LIST = info_collector.info_dict["original_relations"]

    er_aligner = tgtEntRelConstructor(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)

    trainset = jsonDataset(data_train, sentbertnizer, er_aligner)
    testset = jsonDataset(data_test, sentbertnizer, er_aligner)

    trainloader = DataLoader(trainset, batch_size=16, collate_fn=collate_fn)
    testloader = DataLoader(trainset, batch_size=16, collate_fn=collate_fn)

    print("+ Preparing model.")
    embedding_size = sentbertnizer.embedding_size
    model = BERTGraphRel(num_ne=num_ne,
                         num_rel=num_rel,
                         embedding_size=embedding_size,
                         hidden_size=256,
                         n_rnn_layers=2)

    print("+ Start evaluation.")
    eval_BERTGraphRel_model(model=model,
                            trainloader=None,
                            testloader=testloader,
                            device="cpu", # cuda:0 / cpu
                            model_save_path=model_path,
                            load_model=True)

    print("+ done!")

