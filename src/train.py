import os
import glob
import torch
import numpy as np

import albumentations
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine

from model import CaptchaModel
from torch import nn
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp)
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    image_files_png = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    image_files.extend(image_files_png)
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    # abcde -> [a, b, c, d, e]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]  # squeeze

    # Encode Images
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1
    # print(targets_enc)
    # print(len(lbl_enc.classes_))
    # print(np.unique(targets_flat))
    (train_imgs,
     test_imgs,
     train_targets,
     test_targets,
     _,
     test_orig_targets
     ) = model_selection.train_test_split(
        image_files,
        targets_enc,
        targets_orig,
        test_size=0.1,
        random_state=42)

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    def early_stopping(patience, count, prev_loss, current_loss, threshold):
        if abs(prev_loss - current_loss) < threshold and count >= patience:
            return "stop"
        elif abs(prev_loss - current_loss) < threshold:
            return "count"
        else:
            return False

    patience = 6
    count = 0
    prev_train_loss = 0
    threshold = 0.05
    loss = []

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_cap_preds = []

        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)

        pprint(list(zip(test_orig_targets, valid_cap_preds))[15:20])
        print(f"Epoch: {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")

        res = early_stopping(patience, count, prev_train_loss, train_loss, threshold)

        loss.append(train_loss)

        if res == "stop":
            print("Early Stopping Implemented.")
            final_epoch = epoch
            break
        elif res == "count" and train_loss < 0.2:
            count += 1
            print(f"Patience at {patience - count}")
        else:
            prev_train_loss = train_loss

    df_pytorch = pd.DataFrame({"loss": loss})
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Value")
    plt.title("Loss During Epoch Training")
    sns.lineplot(data=df_pytorch, palette="tab10", linewidth=2.5)


if __name__ == "__main__":
    run_training()
    # 75 values: "6'66dddd'''dddd''77'''8'''h"
