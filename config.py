##### config.py #####
import torch

# Paths
TRAIN_DATA_DIR = "/kaggle/input/create-numpy-dataset-exp-name"
TEST_DATA_DIR = "/kaggle/input/czii-cryo-et-object-identification"

# Training Parameters
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
CACHE_RATE = 1.0  

# Model Parameters
CHANNELS = (48, 64, 80, 80)
STRIDES = (2, 2, 1)
NUM_RES_UNITS = 1
OUT_CHANNELS = 7
IN_CHANNELS = 1
SPATIAL_DIMS = 3

# Compute Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

