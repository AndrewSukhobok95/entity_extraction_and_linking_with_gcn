import os
from data_processing.EntityRelationInfoCollector import InfoCollector
from model.BERTGraphRelExtractor import BERTGraphRelExtractor
from data_processing.data_prep import get_dataset

info_dict_json_path = "./../json_dicts/dumb_info.json"
data_json_test_path = "./../data/dumb_2/dumb_test.json"
model_path = "./../trained_models/dumb_bertlgl_v0.pth"

# info_dict_json_path = "./../json_dicts/NYT_info.json"
# data_json_test_path = "./../data/preproc_NYT_json/test.json"
# model_path = "./../trained_models/nyt_bertgl_v0.pth"


data_test, _, _ = get_dataset(data_json_test_path)

def pretty_pred_entity_sent_print(tokens_base, entity_type):
    s = []
    for t, te in zip(tokens_base, entity_type):
        if (t!="[CLS]") & (t!="[SEP]"):
            if te!="O":
                t = t + " (" + te + ")"
            s.append(t)
    print("+++", " ".join(s))

def pretty_pred_entity_print(tokens_base, entity_type):
    for t, te in zip(tokens_base, entity_type):
        if te != "O":
            print("+++", te, "-", t)

def pretty_pred_rel_print(tokens_base, rel_list):
    for r in rel_list:
        print("+++", tokens_base[r[0]], "-", tokens_base[r[1]], "-", r[2])

def pretty_true_entity_print(entityMentions):
    for e in entityMentions:
        print("+++", e["label"], "-", e["text"])

def pretty_true_rel_print(relationMentions):
    for r in relationMentions:
        print("+++", r['em1Text'], "-", r['em2Text'], "-", r["label"])

if __name__=="__main__":

    info_collector = InfoCollector()
    info_collector.load_info_dict(path=info_dict_json_path)
    relextractor = BERTGraphRelExtractor(info_dict=info_collector.info_dict,
                                         trained_model_path=model_path)

    my_sentence = "Sberbank is the largest company in Russia"
    tokens_base, entity_type, rel_list = relextractor.analyze_sentence(my_sentence)
    print("+ Original sentence:", my_sentence)
    print("+ PREDICTED ENTITIES:")
    pretty_pred_entity_print(tokens_base, entity_type)
    pretty_pred_entity_sent_print(tokens_base, entity_type)
    print("+ PREDICTED RELATIONS:")
    pretty_pred_rel_print(tokens_base, rel_list)
    print()
    
    for obs in data_test:

        sentText = obs["sentText"]
        entityMentions = obs["entityMentions"]
        relationMentions = obs["relationMentions"]

        tokens_base, entity_type, rel_list = relextractor.analyze_sentence(sentText)

        print()
        print("+ Original sentence:", sentText)
        print("+ TRUE ENTITIES:")
        pretty_true_entity_print(entityMentions)
        print("+ PREDICTED ENTITIES:")
        pretty_pred_entity_print(tokens_base, entity_type)
        pretty_pred_entity_sent_print(tokens_base, entity_type)

        print("+ TRUE RELATIONS:")
        pretty_true_rel_print(relationMentions)
        print("+ PREDICTED RELATIONS:")
        pretty_pred_rel_print(tokens_base, rel_list)
        print()

    print("+ done!")
