import albumentations
import numpy as np
import pandas as pd
import os
import glob
import random
from pprint import pprint
from tqdm import tqdm
from torch.utils import data
# PyTorch Model
import torch
from torch import nn
from torch.nn import functional as F
# Dataset Loading
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# Model Training
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_utils import *

# Configurations for the files
DIR = "../input/captcha-images/"
BATCH_SIZE = 8
IMG_HEIGHT = 75
IMG_WIDTH = 300
EPOCHS = 150
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

paths = []
labels = []
for image in os.listdir(DIR):
    paths.append(os.path.join(DIR, image))
    labels.append(image.split(".")[0])

df = pd.DataFrame({
    "paths": paths,
    "labels": labels
})

if __name__ == "__main__":
    image_files, targets_enc, targets_orig, lbl_enc = encode_targets()

    (train_imgs, test_imgs, train_targets, test_targets, _, test_orig_targets) = train_test_split(
    image_files, targets_enc, targets_orig, test_size=0.1, random_state=0)

    train_dataset = DatasetClassifier(
    image_paths=train_imgs, targets=train_targets, resize=(IMG_HEIGHT, IMG_WIDTH))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    test_dataset = DatasetClassifier(
        image_paths=test_imgs, targets=test_targets, resize=(IMG_HEIGHT, IMG_WIDTH)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    model = MyCaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(DEVICE)

    # Create optimizer and callbacks
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    # Training
    patience = 6
    count = 0
    prev_train_loss = 0
    threshold = 0.05
    loss = []

    for epoch in range(EPOCHS):
        train_loss = train_function(model, train_loader, optimizer)
        valid_preds, valid_loss = eval_function(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)   
        pprint(list(zip(test_orig_targets, valid_cap_preds))[15:20])
        print(f"Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        res = early_stopping(patience, count, prev_train_loss, train_loss, threshold)
        loss.append(train_loss)
        if res == "stop":
            print("Early Stopping Implemented.")
            final_epoch = epoch
            break
        elif res == "count" and train_loss < 0.2:
            count += 1
            print(f"Patience at {patience-count}")
        else:
            prev_train_loss = train_loss

    torch.save(model.state_dict(), "./model.bin")

    df_pytorch = pd.DataFrame({"loss": loss})
    plt.figure(figsize=(15,5))
    plt.grid()
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss Value")
    plt.title("Loss During Epoch Training")
    sns.lineplot(data=df_pytorch, palette="tab10", linewidth=2.5)
    image_path, eval_loader = get_sample_photo()
    print(image_path)
    preds = predict_captcha(model, eval_loader, image_path)
    print(preds)