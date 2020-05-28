import torch
from typing import List

class EntityConfusionStats(object):
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def getPrecision(self) -> float:
        denom = self.TP + self.FP
        if denom == 0:
            return 0
        p = self.TP / denom
        return p

    def getRecall(self) -> float:
        denom = self.TP + self.FN
        if denom == 0:
            return 0
        r = self.TP / denom
        return r

    def getF1(self) -> float:
        p = self.getPrecision()
        r = self.getRecall()
        if p + r == 0:
            return 0
        f1 = 2 * p * r / (p + r)
        return f1

    def updateStats(self,
                    batch_ne_pred: torch.tensor,
                    batch_ne_true: torch.tensor,
                    sentences_length: List[int]):

        batch_size = batch_ne_pred.size(1)
        batch_ne_pred = batch_ne_pred.argmax(dim=2)

        for b in range(batch_size):
            sl = sentences_length[b]
            sp = batch_ne_pred[:sl, b].detach().numpy()
            st = batch_ne_true[:sl, b].detach().numpy()

            for i in range(sl):
                if (sp[i]==st[i]) & (st[i]==0):
                    self.TP += 1
                elif (sp[i]!=0) & (st[i]==0):
                    self.FP += 1
                elif (sp[i]==0) & (st[i]!=0):
                    self.FN += 1




class TripletConfusionStats(object):
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.true_binary_rel = 0
        self.full_binary_rel = 0

    def getPrecision(self) -> float:
        denom = self.TP + self.FP
        if denom == 0:
            return 0
        p = self.TP / denom
        return p

    def getRecall(self) -> float:
        denom = self.TP + self.FN
        if denom == 0:
            return 0
        r = self.TP / denom
        return r

    def getF1(self) -> float:
        p = self.getPrecision()
        r = self.getRecall()
        if p + r == 0:
            return 0
        f1 = 2 * p * r / (p + r)
        return f1

    def getBinaryRelationsAccuracy(self):
        if self.full_binary_rel==0:
            return 0
        return self.true_binary_rel / self.full_binary_rel

    def updateStats(self,
                    batch_ne_pred: torch.tensor,
                    batch_ne_true: torch.tensor,
                    batch_rel_pred: torch.tensor,
                    batch_rel_true: torch.tensor,
                    sentences_length: List[int]):

        batch_size = batch_ne_pred.size(1)

        batch_ne_pred = batch_ne_pred.argmax(dim=2)
        batch_rel_pred = batch_rel_pred.argmax(dim=3)

        for b in range(batch_size):
            sl = sentences_length[b]
            pred_rel_inds = batch_rel_pred[:sl, :sl, b].nonzero().detach().cpu().numpy()
            true_rel_inds = batch_rel_true[:sl, :sl, b].nonzero().detach().cpu().numpy()

            cur_tp = 0
            cur_br = 0

            for pr in pred_rel_inds:
                for tr in true_rel_inds:
                    y_pr, x_pr = pr
                    y_tr, x_tr = tr

                    pred_rel = batch_rel_pred[y_pr, x_pr, b].item()
                    pred_e1 = batch_ne_pred[y_pr, b].item()
                    pred_e2 = batch_ne_pred[x_pr, b].item()

                    true_rel = batch_rel_true[y_tr, x_tr, b].item()
                    true_e1 = batch_ne_true[y_tr, b].item()
                    true_e2 = batch_ne_true[x_tr, b].item()

                    if (y_pr == y_tr) & (x_pr == x_tr):
                        cur_br += 1

                    if (y_pr == y_tr) & (x_pr == x_tr) & (pred_rel==true_rel) & (pred_e1==true_e1) & (pred_e2==true_e2):
                        cur_tp += 1

            self.TP += cur_tp
            self.FP += pred_rel_inds.shape[0] - cur_tp
            self.FN += true_rel_inds.shape[0] - cur_tp
            self.true_binary_rel += cur_br
            self.full_binary_rel += true_rel_inds.shape[0]




if __name__=="__main__":

    b_ne_tensor_t = torch.tensor([
        [0, 7, 0, 2, 0],
        [0, 2, 0, 6, 0]
    ]).permute(1, 0)
    b_ne_tensor_p = torch.tensor([
        [0, 7, 0, 2, 0],
        [0, 2, 5, 1, 0]
    ]).permute(1, 0)

    b_rel_tensor_t = torch.tensor([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],[
        [0, 0, 0, -1, -1],
        [0, 0, 0, -1, -1],
        [0, 0, 0, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ]]).permute(1, 2, 0)
    b_rel_tensor_p = torch.tensor([[
        [0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],[
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [9, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]]).permute(1, 2, 0)

    sentences_length = [5, 3]

    ecs = EntityConfusionStats()
    ecs.updateStats(b_ne_tensor_p, b_ne_tensor_t, sentences_length)

    tcs = TripletConfusionStats()
    tcs.updateStats(b_ne_tensor_p, b_ne_tensor_t, b_rel_tensor_p, b_rel_tensor_t, sentences_length)

    print(tcs.TP, tcs.FN, tcs.FP)

    print("+ done!")

