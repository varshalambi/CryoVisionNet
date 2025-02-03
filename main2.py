##### training.py #####
import torch
import lightning.pytorch as pl
from config import NUM_EPOCHS, device
from model import Model
from data_loading import load_data, get_dataloaders


def train_model():
    """
    Initialize and train the UNet model using Lightning Trainer.
    """
    train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
    valid_names = ['TS_6_4']
    
    train_files, valid_files = load_data(train_names, valid_names)
    train_loader, valid_loader = get_dataloaders(train_files, valid_files)
    
    model = Model()
    model.to(device)
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("No GPU available. Running on CPU.")
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else None,
        num_nodes=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    train_model()

