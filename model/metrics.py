import torch


class F1:
    def __init__(self):
        self.P = [0, 0]
        self.R = [0, 0]

    def get(self):
        try:
            P = self.P[0] / self.P[1]
        except:
            P = 0

        try:
            R = self.R[0] / self.R[1]
        except:
            R = 0

        try:
            F = 2 * P * R / (P + R)
        except:
            F = 0

        return P, R, F

    def add(self, ro, ra):
        self.P[1] += len(ro)
        self.R[1] += len(ra)

        for r in ro:
            if r in ra:
                self.P[0] += 1

        for r in ra:
            if r in ro:
                self.R[0] += 1


if __name__=="__main__":

    print("+ done!")

