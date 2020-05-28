import torch
import torch.nn as nn
import time

from model.metrics import TripletConfusionStats, EntityConfusionStats


def eval_BERTGraphRel_model(model,
                            trainloader=None,
                            testloader=None,
                            device="cpu",
                            model_save_path="./model.pth",
                            load_model=True):

    if load_model:
        model.load_state_dict(
            torch.load(model_save_path,
                       map_location=lambda storage, loc: storage)
        )
        print("+ Model", model_save_path, "loaded.")

    tdevice = torch.device(device)
    model.to(tdevice)
    model.eval()

    if testloader is not None:
        print("+++ Start evaluation of test set.")
        test_start_time = time.time()
        test_triplet_confusion = TripletConfusionStats()
        test_entity_confusion = EntityConfusionStats()
        for i, batch in enumerate(testloader):

            b_avgemb_tensor, b_ne_tensor, b_rel_tensor, sentences_length = batch

            b_avgemb_tensor = b_avgemb_tensor.to(tdevice)
            b_ne_tensor = b_ne_tensor.to(tdevice)
            b_rel_tensor = b_rel_tensor.to(tdevice)

            out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2 = model(b_avgemb_tensor)

            test_triplet_confusion.updateStats(batch_ne_pred=out_ne_p2,
                                               batch_ne_true=b_ne_tensor,
                                               batch_rel_pred=out_rel_p2,
                                               batch_rel_true=b_rel_tensor,
                                               sentences_length=sentences_length)
            test_entity_confusion.updateStats(batch_ne_pred=out_ne_p2,
                                              batch_ne_true=b_ne_tensor,
                                              sentences_length=sentences_length)
            if i % 50 == 0:
                print("iter =", i,
                      "rP = {0:0.3f}".format(test_triplet_confusion.getPrecision()),
                      "rR = {0:0.3f}".format(test_triplet_confusion.getRecall()),
                      "rBinAcc = {0:0.3f}".format(test_triplet_confusion.getBinaryRelationsAccuracy()),
                      "eP = {0:0.3f}".format(test_entity_confusion.getPrecision()),
                      "eR = {0:0.3f}".format(test_entity_confusion.getRecall()),
                      sep=' ', end='\r')

        test_rel_precision = test_triplet_confusion.getPrecision()
        test_rel_recall = test_triplet_confusion.getRecall()
        test_rel_F1 = test_triplet_confusion.getF1()
        test_rel_binary_acc = test_triplet_confusion.getBinaryRelationsAccuracy()
        test_ner_precision = test_entity_confusion.getPrecision()
        test_ner_recall = test_entity_confusion.getRecall()
        test_ner_F1 = test_entity_confusion.getF1()

        dur = (int)(time.time() - test_start_time)
        print("+++ Evaluation on test set finished in {0:d}m:{1:d}s".format(dur // 60, dur % 60))
        print("+++ Triplet stats:")
        print("+++   Precision:", test_rel_precision)
        print("+++   Recall:", test_rel_recall)
        print("+++   F1:", test_rel_F1)
        print("+++   Binary accuracy:", test_rel_binary_acc)
        print("+++ NER stats:")
        print("+++   Precision:", test_ner_precision)
        print("+++   Recall:", test_ner_recall)
        print("+++   F1:", test_ner_F1)

    if trainloader is not None:
        print("+++ Start evaluation of train set.")
        train_start_time = time.time()
        train_triplet_confusion = TripletConfusionStats()
        train_entity_confusion = EntityConfusionStats()
        for i, batch in enumerate(trainloader):
            b_avgemb_tensor, b_ne_tensor, b_rel_tensor, sentences_length = batch

            b_avgemb_tensor = b_avgemb_tensor.to(tdevice)
            b_ne_tensor = b_ne_tensor.to(tdevice)
            b_rel_tensor = b_rel_tensor.to(tdevice)

            out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2 = model(b_avgemb_tensor)

            train_triplet_confusion.updateStats(batch_ne_pred=out_ne_p2,
                                                batch_ne_true=b_ne_tensor,
                                                batch_rel_pred=out_rel_p2,
                                                batch_rel_true=b_rel_tensor,
                                                sentences_length=sentences_length)

            if i % 50 == 0:
                print("iter =", i,
                      "rP = {0:0.3f}".format(train_triplet_confusion.getPrecision()),
                      "rR = {0:0.3f}".format(train_triplet_confusion.getRecall()),
                      "rBinAcc = {0:0.3f}".format(train_triplet_confusion.getBinaryRelationsAccuracy()),
                      "eP = {0:0.3f}".format(train_entity_confusion.getPrecision()),
                      "eR = {0:0.3f}".format(train_entity_confusion.getRecall()),
                      sep=' ', end='\r')

        train_rel_precision = train_triplet_confusion.getPrecision()
        train_rel_recall = train_triplet_confusion.getRecall()
        train_rel_F1 = train_triplet_confusion.getF1()
        train_rel_binary_acc = train_triplet_confusion.getBinaryRelationsAccuracy()
        train_ner_precision = train_entity_confusion.getPrecision()
        train_ner_recall = train_entity_confusion.getRecall()
        train_ner_F1 = train_entity_confusion.getF1()

        dur = (int)(time.time() - train_start_time)
        print("+++ Evaluation on train set finished in {0:d}m:{1:d}s".format(dur // 60, dur % 60))
        print("+++ Triplet stats:")
        print("+++   Precision:", train_rel_precision)
        print("+++   Recall:", train_rel_recall)
        print("+++   F1:", train_rel_F1)
        print("+++   Binary accuracy:", train_rel_binary_acc)
        print("+++ NER stats:")
        print("+++   Precision:", train_ner_precision)
        print("+++   Recall:", train_ner_recall)
        print("+++   F1:", train_ner_F1)

if __name__=="__main__":

    print("+ done!")



