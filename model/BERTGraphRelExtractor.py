import torch

from data_processing.BERTinizer import SentenceBERTinizer
from data_processing.EntityRelationInfoCollector import InfoCollector
from model.BERTGraphRel import BERTGraphRel

class BERTGraphRelExtractor(object):
    def __init__(self, info_dict, trained_model_path=None):
        self.info_dict = info_dict
        self.bertinizer = SentenceBERTinizer()
        self.model = BERTGraphRel(num_ne=info_dict["entity_vsize"],
                                  num_rel=info_dict["rel_vsize"],
                                  embedding_size=self.bertinizer.embedding_size)
        if trained_model_path is not None:
            self.load_trained_model(trained_model_path)

    def load_trained_model(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage))
        print("Model", path, "loaded.")

    def get_entities(self, entity_pred_tensor):
        d_id_to_type = self.info_dict["mod_entities_id_to_token_dict"]
        entity_index = entity_pred_tensor.squeeze().argmax(dim=1)
        entity_index_np = entity_index.detach().numpy()
        entity_type = [d_id_to_type[str(i)] for i in entity_index_np]
        return entity_type

    def get_relations(self, rel_pred_tensor):
        rel_list = []
        d_id_to_type = self.info_dict["mod_relations_id_to_token_dict"]
        rel_matrix_index = rel_pred_tensor.squeeze(dim=2).argmax(dim=2)
        rel_matrix_index_np = rel_matrix_index.detach().numpy()
        e1_index_array, e2_index_array = rel_matrix_index_np.nonzero()
        if len(e1_index_array)==0:
            return rel_list
        for i in range(len(e1_index_array)):
            e1_index = e1_index_array[i]
            e2_index = e2_index_array[i]
            rel_index = rel_matrix_index_np[e1_index, e2_index]
            rel_list.append([e1_index, e2_index, d_id_to_type[str(rel_index)]])
        return rel_list

    def analyze_sentence(self, sentence):
        tokens_wp, tokens_ids_wp_tensor, segments_ids_wp_tensors, tokens_base = self.bertinizer.tokenize(sentence)
        bert_embeddings = self.bertinizer.get_embeddings(tokens_ids_wp_tensor, segments_ids_wp_tensors)
        bert_avg_embeddings = self.bertinizer.average_wp_embeddings(bert_embeddings, tokens_wp)
        ne_p1, rel_p1, ne_p2, rel_p2 = self.model(bert_avg_embeddings.unsqueeze(dim=1))
        entity_type = self.get_entities(ne_p2)
        rel_list = self.get_relations(rel_p2)
        return tokens_base, entity_type, rel_list


if __name__=="__main__":

    info_collector = InfoCollector()
    info_collector.load_info_dict(path="../json_dicts/NYT_info.json")
    relextractor = BERTGraphRelExtractor(info_dict=info_collector.info_dict,
                                         trained_model_path="../bertgl_v0_zombie.pth")

    '''
     "relationMentions": [{"em1Text": "Ben Nelson", "em2Text": "Nebraska", "label": "/people/person/place_lived"},
                          {"em1Text": "Nebraska", "em2Text": "Ben Nelson", "label": "None"}],
     "entityMentions": [{"start": 0, "text": "Ben Nelson", "label": "PERSON"},
                        {"start": 1, "text": "Nebraska", "label": "LOCATION"}],
    '''
    s = "\"The American people can see what is happening here , '' said Senator Ben Nelson , Democrat of Nebraska . ''\"\r\n"

    relextractor.analyze_sentence(s)

    print("+ done!")
