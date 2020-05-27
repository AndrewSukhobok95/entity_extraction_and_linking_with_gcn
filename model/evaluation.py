import torch
import torch.nn as nn
import time

from model.metrics import TripletConfusionStats


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
            if i % 50 == 0:
                print("P = {0:0.3f}".format(test_triplet_confusion.getPrecision()),
                      "R = {0:0.3f}".format(test_triplet_confusion.getRecall()),
                      sep=' ', end='\r')

        test_precision = test_triplet_confusion.getPrecision()
        test_recall = test_triplet_confusion.getRecall()
        test_F1 = test_triplet_confusion.getF1()

        dur = (int)(time.time() - test_start_time)
        print("+++ Evaluation on test set finished in {0:d}m:{1:d}s".format(dur // 60, dur % 60))
        print("+++ Precision on test set:", test_precision)
        print("+++ Recall on test set:", test_recall)
        print("+++ F1 on test set:", test_F1)

    if trainloader is not None:
        print("+++ Start evaluation of train set.")
        train_start_time = time.time()
        train_triplet_confusion = TripletConfusionStats()
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
                print("P = {0:0.3f}".format(train_triplet_confusion.getPrecision()),
                      "R = {0:0.3f}".format(train_triplet_confusion.getRecall()),
                      sep=' ', end='\r')

        train_precision = train_triplet_confusion.getPrecision()
        train_recall = train_triplet_confusion.getRecall()
        train_F1 = train_triplet_confusion.getF1()

        dur = (int)(time.time() - train_start_time)
        print("+++ Evaluation on train set finished in {0:d}m:{1:d}s".format(dur // 60, dur % 60))
        print("+++ Precision on train set:", train_precision)
        print("+++ Recall on train set:", train_recall)
        print("+++ F1 on train set:", train_F1)

if __name__=="__main__":

    print("+ done!")



