import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel

from data_processing.BERTinizer import SentenceBERTinizer
from data_processing.data_prep import EntityRelationsAligner, get_dataset
from data_processing.data_load import NYTjsonDataset, collate_fn
from model.BERTGraphRel import BERTGraphRel
from model.training import train_BERTGraphRel_model



if __name__=="__main__":

    # Data Prep

    print("+ Reading data.")

    _bert_wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    nyt_json_train = "./data/preproc_NYT_json/train.json"
    nyt_json_test = "./data/preproc_NYT_json/test.json"

    data_nyt_train, NE_LIST, REL_LIST = get_dataset(nyt_json_train, _bert_wp_tokenizer)
    data_nyt_test, _, _ = get_dataset(nyt_json_test, _bert_wp_tokenizer)

    for d in data_nyt_train:
        l = len(_bert_wp_tokenizer.wordpiece_tokenizer.tokenize(d["sentText"]))
        if l > 512:
            print("found")

    print("+ Preparing data.")

    sentbertnizer = SentenceBERTinizer()
    er_aligner = EntityRelationsAligner(tokenizer=sentbertnizer, ne_tags=NE_LIST, rel_tags=REL_LIST)

    trainset = NYTjsonDataset(data_nyt_train, sentbertnizer, er_aligner)
    testset = NYTjsonDataset(data_nyt_test, sentbertnizer, er_aligner)

    trainloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)
    testloader = DataLoader(trainset, batch_size=8, collate_fn=collate_fn)

    # Model Prep

    print("+ Preparing model.")

    num_ne = er_aligner.NE_vsize
    num_rel = er_aligner.REL_vsize
    embedding_size = sentbertnizer.embedding_size

    model = BERTGraphRel(num_ne=num_ne,
                         num_rel=num_rel,
                         embedding_size=embedding_size,
                         hidden_size=256,
                         n_rnn_layers=2)

    # Model training

    print("+ Start training.")

    train_BERTGraphRel_model(model=model,
                             trainloader=trainloader,
                             testloader=testloader,
                             device="cuda:0", # cuda:0 / cpu
                             model_save_path="./bertgl_v0.pth",
                             nepochs=50,
                             lr=0.0001,
                             loss_p2_weight=2)

    print("+ done!")

