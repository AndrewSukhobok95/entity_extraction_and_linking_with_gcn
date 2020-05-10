import torch
import torch.nn as nn
import time


def loss_calc(criteriation, loss_p2_weight, pred_ne_p1, pred_ne_p2, pred_rel_p1, pred_rel_p2, tgt_ne, tgt_rel):
    loss_ne_p1 = criteriation(pred_ne_p1, tgt_ne)
    loss_ne_p2 = criteriation(pred_ne_p2, tgt_ne)

    loss_rel_p1 = criteriation(pred_rel_p1, tgt_rel)
    loss_rel_p2 = criteriation(pred_rel_p2, tgt_rel)

    loss = (loss_ne_p1 + loss_rel_p1) + loss_p2_weight * (loss_ne_p2 + loss_rel_p2)
    return loss


def train_BERTGraphRel_model(model,
                             trainloader,
                             testloader=None,
                             device="cpu",
                             model_save_path="./model.pth",
                             nepochs=50,
                             lr=0.0001,
                             loss_p2_weight=2,
                             loss_ignore_index=-1):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criteriation = nn.NLLLoss(ignore_index=loss_ignore_index)

    tdevice = torch.device(device)
    model.to(tdevice)

    train_start_time = time.time()

    for epoch in range(nepochs):
        model.train()
        for i, batch in enumerate(trainloader):
            b_avgemb_tensor, b_ne_tensor, b_rel_tensor = batch

            b_avgemb_tensor = b_avgemb_tensor.to(tdevice)
            b_ne_tensor = b_ne_tensor.to(tdevice)
            b_rel_tensor = b_rel_tensor.to(tdevice)

            out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2 = model(b_avgemb_tensor)

            # dim ( seq_len, batch, num_ne ) -> ( batch, num_ne, seq_len )
            pred_ne_p1 = out_ne_p1.permute(1, 2, 0)

            pred_ne_p2 = out_ne_p2.permute(1, 2, 0)
            # dim ( seq_len, seq_len, batch, num_rel ) -> ( batch, num_rel, seq_len, seq_len )
            pred_rel_p1 = out_rel_p1.permute(2, 3, 0, 1)
            pred_rel_p2 = out_rel_p2.permute(2, 3, 0, 1)

            # dim ( seq_len, batch ) -> ( batch, seq_len )
            tgt_ne = b_ne_tensor.permute(1, 0).long()
            # dim ( seq_len, seq_len, batch ) -> ( batch, seq_len, seq_len )
            tgt_rel = b_rel_tensor.permute(2, 0, 1).long()

            loss = loss_calc(criteriation,
                             loss_p2_weight,
                             pred_ne_p1,
                             pred_ne_p2,
                             pred_rel_p1,
                             pred_rel_p2,
                             tgt_ne,
                             tgt_rel)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                dur = (int)(time.time() - train_start_time)
                print("{0:d} batches done in {1:d}m:{2:d}s".format(i + 1, dur // 60, dur % 60), end='\r')
                torch.save(model.state_dict(), model_save_path)

        torch.save(model.state_dict(), model_save_path)

        print("+ Epoch {0:d}: Train Loss:\t{1:0.3f}".format(epoch + 1, loss.item()))

        if testloader is not None:
            model.eval()
            for i, batch in enumerate(testloader):
                b_avgemb_tensor, b_ne_tensor, b_rel_tensor = batch

                b_avgemb_tensor = b_avgemb_tensor.to(tdevice)
                b_ne_tensor = b_ne_tensor.to(tdevice)
                b_rel_tensor = b_rel_tensor.to(tdevice)

                out_ne_p1, out_rel_p1, out_ne_p2, out_rel_p2 = model(b_avgemb_tensor)

                # dim ( seq_len, batch, num_ne ) -> ( batch, num_ne, seq_len )
                pred_ne_p1 = out_ne_p1.permute(1, 2, 0)

                pred_ne_p2 = out_ne_p2.permute(1, 2, 0)
                # dim ( seq_len, seq_len, batch, num_rel ) -> ( batch, num_rel, seq_len, seq_len )
                pred_rel_p1 = out_rel_p1.permute(2, 3, 0, 1)
                pred_rel_p2 = out_rel_p2.permute(2, 3, 0, 1)

                # dim ( seq_len, batch ) -> ( batch, seq_len )
                tgt_ne = b_ne_tensor.permute(1, 0).long()
                # dim ( seq_len, seq_len, batch ) -> ( batch, seq_len, seq_len )
                tgt_rel = b_rel_tensor.permute(2, 0, 1).long()

                with torch.no_grad():
                    test_loss = loss_calc(criteriation,
                                          loss_p2_weight,
                                          pred_ne_p1,
                                          pred_ne_p2,
                                          pred_rel_p1,
                                          pred_rel_p2,
                                          tgt_ne,
                                          tgt_rel)

            print("+ Epoch {0:d}: Test Loss:\t{1:0.3f}".format(epoch + 1, test_loss.item()))

        print("=================================")

if __name__=="__main__":

    print("+ done!")
