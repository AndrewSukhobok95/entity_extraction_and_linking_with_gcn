import json
import re
import numpy as np
import random


# {"sentId": 33, "articleId": "2",
#  "relationMentions": [
#      {"em1Text": "Tim Pawlenty", "em2Text": "Minnesota", "label": "/people/person/place_lived"},
#      {"em1Text": "Minnesota", "em2Text": "Tim Pawlenty", "label": "None"}
#  ],
#  "entityMentions": [
#      {"start": 0, "text": "Tim Pawlenty", "label": "PERSON"},
#      {"start": 1, "text": "Minnesota", "label": "LOCATION"}
#  ],
#  "sentText": "Gov. Tim Pawlenty of Minnesota ordered the state health department this month to monitor day-to-day operations at the Minneapolis Veterans Home after state inspectors found that three men had died there in the previous month because of neglect or medical errors .\r\n"}


class DumbDataSetConstructor(object):
    def __init__(self, config_path):
        with open(config_path) as json_file:
            self.config_json = json.load(json_file)
        self.entity_list = self.config_json["entity_types"]
        self.relation_list = list(self.config_json["relation_sentences"].keys())
        self.sample = []

    def get_rand_entity(self, entity_name):
        entity_name_short = re.sub(r'\[|\]|\{\d+\}', "", entity_name)
        entity_type = self.config_json["entities"][entity_name_short]["type"]
        entity_text = random.choice(self.config_json["entities"][entity_name_short]["values"])
        return entity_type, entity_text

    def generate_rel_sample(self, rel_type):
        sample = {}
        entityMentions = []
        relationMentions = []

        rel_sentences = self.config_json["relation_sentences"][rel_type]
        s = random.choice(rel_sentences)
        sentText = s[0]

        s_entities = re.findall(r'\[[A-Z]+\]\{\d+\}|\[[A-Z]+\]', sentText)
        e_text_dict = {}
        for e in s_entities:
            entity_type, entity_text = self.get_rand_entity(e)
            if entity_type is not None:
                entityMentions.append({"text": entity_text, "label": entity_type})
            e_text_dict[e] = entity_text
        sample["entityMentions"] = entityMentions

        for r in s[1]:
            relationMentions.append({
                "em1Text": e_text_dict[r[0]],
                "em2Text": e_text_dict[r[1]],
                "label": rel_type
            })
        sample["relationMentions"] = relationMentions

        for k in e_text_dict.keys():
            sentText = sentText.replace(k, e_text_dict[k])
        sample["sentText"] = sentText

        return sample

    def generate_dataset(self, n):
        n_rels = len(self.relation_list)
        n_samples_per_rel = n // n_rels
        n_samples_left = n - n_samples_per_rel * n_rels
        for r in self.relation_list:
            for i in range(n_samples_per_rel):
                s = self.generate_rel_sample(r)
                self.sample.append(s)
        for i in range(n_samples_left):
            s = self.generate_rel_sample(r)
            self.sample.append(s)

    def clear_sample(self):
        self.sample = []

    def get_dataset(self):
        return self.sample, self.entity_list, self.relation_list

    def write_json_dataset(self, path):
        with open(path, 'a') as json_file:
            for s in self.sample:
                json.dump(s, json_file)
                json_file.write("\n")



if __name__=="__main__":

    ddsc = DumbDataSetConstructor(config_path="./../json_dicts/dumb_dataset_parts.json")
    # ddsc.generate_rel_sample("work_in")

    ddsc.generate_dataset(n=100)
    data, ne_list, rel_list = ddsc.get_dataset()

    print("+ done!")

