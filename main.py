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

train = False
eval = True
save_text_info = False
load_text_info_from_json = True
text_info_dict_path = "./json_dicts/NYT_info.json"
nyt_json_train = "./data/preproc_NYT_json/train.json"
nyt_json_test = "./data/preproc_NYT_json/test.json"
model_path = "./trained_models/nyt_bertgl_v0.pth"

if __name__=="__main__":

    # BERT Prep

    print("+ Preparing BERT model.")

    sentbertnizer = SentenceBERTinizer()

    # Data Prep

    print("+ Reading data.")

    data_nyt_train, NE_LIST, REL_LIST = get_dataset(nyt_json_train, sentbertnizer.tokenizer)
    data_nyt_test, _, _ = get_dataset(nyt_json_test, sentbertnizer.tokenizer)

    print("+ Preparing data.")

    info_collector = InfoCollector()

    if load_text_info_from_json:
        print("+ Loading text info.")
        info_collector.load_info_dict(path=text_info_dict_path)
        num_ne = info_collector.info_dict["entity_vsize"]
        num_rel = info_collector.info_dict["rel_vsize"]
        NE_LIST = info_collector.info_dict["original_entities"]
        REL_LIST = info_collector.info_dict["original_relations"]

    er_aligner = tgtEntRelConstructor(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)
    if not load_text_info_from_json:
        num_ne = er_aligner.NE_vsize
        num_rel = er_aligner.REL_vsize

    trainset = jsonDataset(data_nyt_train, sentbertnizer, er_aligner)
    testset = jsonDataset(data_nyt_test, sentbertnizer, er_aligner)

    trainloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)
    testloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)

    # Model Prep
    embedding_size = sentbertnizer.embedding_size

    if save_text_info:
        print("+ Saving text info.")
        info_collector.remember_info(entities=NE_LIST,
                                     relations=REL_LIST,
                                     entity_vsize=num_ne,
                                     rel_vsize=num_rel,
                                     mod_entities=er_aligner.NE_biotags,
                                     mod_relations=er_aligner.REL_mod_tags,
                                     mod_entities_id_dict=er_aligner.NE_bio_to_index_dict,
                                     mod_relations_id_dict=er_aligner.REL_mod_to_index_dict)
        info_collector.save_info_dict(path=text_info_dict_path)

    print("+ Preparing model.")

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
                                 model_save_path=model_path,
                                 nepochs=50,
                                 lr=0.0001,
                                 loss_p2_weight=2,
                                 load_model=True)

    if eval:
        print("+ Start evaluation.")
        eval_BERTGraphRel_model(model=model,
                                trainloader=trainloader,
                                testloader=testloader,
                                device="cuda:0", # cuda:0 / cpu
                                model_save_path=model_path,
                                load_model=True)

    print("+ done!")

