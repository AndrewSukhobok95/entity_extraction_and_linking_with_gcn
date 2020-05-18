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

train = True
eval = True
text_info_dict_path = "./json_dicts/dumb_info.json"
data_json_train_path = "./data/dumb_2/dumb_train.json"
data_json_test_path = "./data/dumb_2/dumb_test.json"

if __name__=="__main__":

    # BERT Prep

    print("+ Preparing BERT model.")

    sentbertnizer = SentenceBERTinizer()

    # Data Prep

    print("+ Reading data.")

    data_train, _, _ = get_dataset(data_json_train_path, sentbertnizer.tokenizer)
    data_test, _, _ = get_dataset(data_json_test_path, sentbertnizer.tokenizer)

    print("+ Loading text info.")
    info_collector = InfoCollector()
    info_collector.load_info_dict(path=text_info_dict_path)
    num_ne = info_collector.info_dict["entity_vsize"]
    num_rel = info_collector.info_dict["rel_vsize"]
    NE_LIST = info_collector.info_dict["original_entities"]
    REL_LIST = info_collector.info_dict["original_relations"]

    print("+ Preparing data.")
    er_aligner = tgtEntRelConstructor(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)

    trainset = jsonDataset(data_train, sentbertnizer, er_aligner)
    testset = jsonDataset(data_test, sentbertnizer, er_aligner)

    trainloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)
    testloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)

    # Model Prep
    print("+ Preparing model.")

    embedding_size = sentbertnizer.embedding_size

    model = BERTGraphRel(num_ne=num_ne,
                         num_rel=num_rel,
                         embedding_size=embedding_size,
                         hidden_size=256,
                         n_rnn_layers=2)

    # Model training

    if train:
        print("+ Start training.")
        train_BERTGraphRel_model(model=model,
                                 trainloader=trainloader,
                                 testloader=None,
                                 device="cuda:0", # cuda:0 / cpu
                                 model_save_path="./bertgl_v1_dumb_haifu.pth",
                                 nepochs=30,
                                 lr=0.0001,
                                 loss_p2_weight=2,
                                 load_model=False)

    if eval:
        print("+ Start evaluation.")
        eval_BERTGraphRel_model(model=model,
                                trainloader=trainloader,
                                testloader=testloader,
                                device="cuda:0", # cuda:0 / cpu
                                model_save_path="./bertgl_v1_dumb_haifu.pth",
                                load_model=True)

    print("+ done!")

