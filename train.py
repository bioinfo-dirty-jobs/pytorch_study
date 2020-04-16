from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcor
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms




for epoch in range(EPOCHS):
      
        for j, data in enumerate(loader_train):
            global_i += 1

            if j % 10 == 0:
                print(time.time() - start_time)
                start_time = time.time()

            optimizer.zero_grad()

            images_tr = data["data"].to(device)
            labels_tr = torch.LongTensor(data["label"]).to(device)
            outputs_tr = model(images_tr).to(device)

            # backward
            loss = criterion(outputs_tr, labels_tr)
            loss.backward()

            optimizer.step()

            # check test set
            if j % int(len(loader_train) / 2) == 0 and j != 0:
                model.eval()
                with torch.no_grad():

                    losses_sum = 0
                    num_samples_test = 0

                    for data_test in loader_test:

                        images_ts = data_test["data"].to(device)
                        labels_ts = torch.LongTensor(data["label"]).to(device)

                        outputs_ts = model.forward(images_ts)

                        loss_test_sum = criterion(outputs_ts, labels_ts).item()
                        losses_sum += loss_test_sum
                        num_samples_test += 1

                    loss_test_avg = losses_sum / num_samples_test
                    mean_loss_train = losses_sum / (
                        len(loader_train) * loader_train.batch_size
                    )
                    

                    last_loss_test = loss_test_avg

                losses_tr.append(loss.item())
                losses_ts.append(loss_test_avg)

                del images_ts, labels_ts

            iteration += 1
            del images_tr, labels_tr
            gc.collect()
            model.train()

         

            sys.stdout.write(
                "\r Epoch {} of {}  [{:.2f}%] - loss TR/TS: {:.4f} / {:.4f} ".format(
                    epoch + 1,
                    EPOCHS,
                    100 * j / len(loader_train),
                    loss.item(),
                    last_loss_test,
                    #  optimizer.param_groups[0]["lr"],
                )
            )
        # torch.backends.cuda.cufft_plan_cache.clear()

    # save losses
    losses_tr = np.array(losses_tr)
    losses_vl = np.array(losses_ts)
    
    # Prediction on TRAINING
    model.eval()

    preds_tr = []
    trues_tr = []
    probs_tr = []
    filenames_tr = []

    with torch.no_grad():
        for data in loader_train:
            image = data["data"].to(device)
            label = data["label"]
            output = model(image)  # forward
            _, pred = torch.max(output, 1)

            preds_tr.append(pred.data.cpu().numpy())
            #         trues.append(label)
            trues_tr.append(data["label"])
            probs_tr.append(output.data.cpu().numpy())
            filenames_tr.append(data["filename"])

    probs_tr = np.concatenate(probs_tr)
    preds_tr = np.concatenate(preds_tr)
    trues_tr = np.concatenate(trues_tr)
    filenames_tr = np.concatenate(filenames_tr)

    MCC_tr = mcor(trues_tr, preds_tr)
    ACC_tr = acc(trues_tr, preds_tr)
    prec_tr = precision(trues_tr, preds_tr, average="weighted")
    rec_tr = recall(trues_tr, preds_tr, average="weighted")

    
    # PREDICTION ON TEST
    model.eval()

    preds_ts = []
    trues_ts = []
    probs_ts = []
    filenames_ts = []

    with torch.no_grad():
        for data in loader_test:
            image = data["data"].to(device)
            label = data["label"]
            output = model(image)  # forward
            _, pred = torch.max(output, 1)

            preds_ts.append(pred.data.cpu().numpy())
            trues_ts.append(data["label"])
            probs_ts.append(output.data.cpu().numpy())
            filenames_ts.append(data["filename"])

    probs_ts = np.concatenate(probs_ts)
    preds_ts = np.concatenate(preds_ts)
    trues_ts = np.concatenate(trues_ts)
    filenames_ts = np.concatenate(filenames_ts)

    MCC_ts = mcor(trues_ts, preds_ts)
    ACC_ts = acc(trues_ts, preds_ts)
    prec_ts = precision(trues_ts, preds_ts, average="weighted")
    rec_ts = recall(trues_ts, preds_ts, average="weighted")

