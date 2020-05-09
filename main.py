import os
import json
import numpy as np

from data_processing.BERTinizer import SentenceBERTinizer
from data_processing.data_prep import EntityRelationsAligner, get_dataset
from model.BERTGraphRel import BERTGraphRel



if __name__=="__main__":

    nyt_json_train = "./../data/preproc_NYT_json/train.json"
    nyt_json_test = "./../data/preproc_NYT_json/test.json"

    data_nyt_train, NE_LIST, REL_LIST = get_dataset(nyt_json_train)
    data_nyt_test, _, _ = get_dataset(nyt_json_test)

    print("+ done!")

