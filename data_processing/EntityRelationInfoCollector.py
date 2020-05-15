import os
import json

class InfoCollector(object):
    def __init__(self):
        self.info_dict = {}

    def remember_original_entities(self, entities):
        self.info_dict["original_entities"] = entities

    def remember_original_relations(self, relations):
        self.info_dict["original_relations"] = relations

    def remember_entity_vector_size(self, entity_vsize):
        self.info_dict["entity_vsize"] = entity_vsize

    def remember_rel_vector_size(self, rel_vsize):
        self.info_dict["rel_vsize"] = rel_vsize

    def remember_mod_entities(self, mod_entities):
        self.info_dict["mod_entities"] = mod_entities

    def remember_mod_relations(self, mod_relations):
        self.info_dict["mod_relations"] = mod_relations

    def remember_mod_entities_id_dict(self, mod_entities_id_dict):
        self.info_dict["mod_entities_token_to_id_dict"] = mod_entities_id_dict
        self.info_dict["mod_entities_id_to_token_dict"] = dict([[v, k] for k, v in mod_entities_id_dict.items()])

    def remember_mod_relations_id_dict(self, mod_relations_id_dict):
        self.info_dict["mod_relations_token_to_id_dict"] = mod_relations_id_dict
        self.info_dict["mod_relations_id_to_token_dict"] = dict([[v, k] for k, v in mod_relations_id_dict.items()])

    def remember_info(self, entities, relations,
                      entity_vsize, rel_vsize,
                      mod_entities, mod_relations,
                      mod_entities_id_dict, mod_relations_id_dict):
        self.remember_original_entities(entities)
        self.remember_original_relations(relations)
        self.remember_entity_vector_size(entity_vsize)
        self.remember_rel_vector_size(rel_vsize)
        self.remember_mod_entities(mod_entities)
        self.remember_mod_relations(mod_relations)
        self.remember_mod_entities_id_dict(mod_entities_id_dict)
        self.remember_mod_relations_id_dict(mod_relations_id_dict)

    def load_info_dict(self, path):
        with open(path) as json_file:
            self.info_dict = json.load(json_file)

    def save_info_dict(self, path):
        with open(path, 'w') as json_file:
            json.dump(self.info_dict, json_file)


if __name__=="__main__":

    info_collector = InfoCollector()
    info_collector.load_info_dict(path="./../json_dicts/NYT_info.json")

    print("+ done!")
