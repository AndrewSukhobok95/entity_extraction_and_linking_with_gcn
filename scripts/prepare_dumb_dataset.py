import os

from data_processing.BERTinizer import SentenceBERTinizer
from data_processing.data_prep import tgtEntRelConstructor
from model_testing.DumbDataSetConstructor import DumbDataSetConstructor
from data_processing.EntityRelationInfoCollector import InfoCollector

dumb_dataset_config_path = "./../json_dicts/dumb_dataset_parts.json"
text_info_dict_path = "./../json_dicts/dumb_info.json"

dumb_dataset_dir = "./../data/dumb/"
if not os.path.exists(dumb_dataset_dir):
    os.makedirs(dumb_dataset_dir)

train_dumb_dataset_path = dumb_dataset_dir + "dumb_train.json"
test_dumb_dataset_path = dumb_dataset_dir + "dumb_test.json"

if __name__=="__main__":

    ddsc = DumbDataSetConstructor(config_path=dumb_dataset_config_path)

    print("+ Preparing and saving train dataset.")
    ddsc.generate_dataset(n=6000)
    data, ne_list, rel_list = ddsc.get_dataset()
    ddsc.write_json_dataset(train_dumb_dataset_path)

    print("+ Preparing and saving descriptive json.")
    sentbertnizer = SentenceBERTinizer()
    er_aligner = tgtEntRelConstructor(tokenizer=sentbertnizer, ne_tags=ne_list, rel_tags=rel_list)

    num_ne = er_aligner.NE_vsize
    num_rel = er_aligner.REL_vsize

    info_collector = InfoCollector()
    info_collector.remember_info(entities=ne_list,
                                 relations=rel_list,
                                 entity_vsize=num_ne,
                                 rel_vsize=num_rel,
                                 mod_entities=er_aligner.NE_biotags,
                                 mod_relations=er_aligner.REL_mod_tags,
                                 mod_entities_id_dict=er_aligner.NE_bio_to_index_dict,
                                 mod_relations_id_dict=er_aligner.REL_mod_to_index_dict)
    info_collector.save_info_dict(path=text_info_dict_path)

    print("+ Preparing and saving test dataset.")
    ddsc.clear_sample()
    ddsc.generate_dataset(n=300)
    ddsc.write_json_dataset(test_dumb_dataset_path)

    print("+ done!")
