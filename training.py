##### training.py #####
import torch
import lightning.pytorch as pl
from config import NUM_EPOCHS, device
from model import Model
from data_loading import load_data, get_dataloaders

def train():
    train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6']
    valid_names = ['TS_6_4']
    train_files, valid_files = load_data(train_names, valid_names)
    train_loader, valid_loader = get_dataloaders(train_files, valid_files)

    model = Model().to(device)

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=[0])
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    train()

