import psutil
import os

DATA_PATH=os.path.join("Task01_BrainTumour")
OUT_PATH = os.path.join("output/")
INFERENCE_FILENAME = "2d_unet_decathlon"

EPOCHS = 20  # Number of epochs to train

BATCH_SIZE = 128

# Using Adam optimizer
LEARNING_RATE = 0.0001  # 0.00005
WEIGHT_DICE_LOSS = 0.85  # Combined loss weight for dice versus BCE

FEATURE_MAPS = 16
PRINT_MODEL = True  # Print the model

BLOCKTIME = 0
NUM_INTER_THREADS = 1

import multiprocessing
NUM_INTRA_THREADS = psutil.cpu_count(logical=False)

CROP_DIM=128  # Crop height and width to this size
SEED=816      # Random seed
TRAIN_TEST_SPLIT=0.80 # The train/test split

CHANNELS_FIRST = False
USE_UPSAMPLING = False
USE_AUGMENTATION = True  # Use data augmentation during training
USE_DROPOUT = True  # Use spatial dropout in model
USE_PCONV = False
