import os
import yaml

# -----------------------------------------------------------------------------
# 1. Data path configuration
# -----------------------------------------------------------------------------
# [Annotation Paths] Update these if you move to a new machine.
TRAIN_ANNO_DIR = '/media/SSD/data/CVPR_workshop/annotation/ABAW Annotations/ABAW Annotations/EXPR_Recognition_Challenge/Train_Set'
VAL_ANNO_DIR = '/media/SSD/data/CVPR_workshop/annotation/ABAW Annotations/ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set'

# [Image Paths] Update these if you move to a new machine.
IMAGE_BASE_DIRS = [
    '/media/SSD/data/CVPR_workshop/cropped_aligned_image/image'
]

# [Audio Path] Keep this consistent with `txt.py`.
AUDIO_DIR = '/media/SSD/data/CVPR_workshop/audio'

# -----------------------------------------------------------------------------
# 2. Hyperparameters
# -----------------------------------------------------------------------------
SPLIT_RATIO = 0.8

# [Caution] This must match the sequence length used by the trained weights (`best_model_vgcmf.pth`).
SEQ_LEN = 90           
STRIDE = 15

# Lowered to 32 to avoid VRAM overflow in a multi-GPU setup (2x RTX 3090). Adjust if needed.
BATCH_SIZE = 32    
GRAD_ACCUM_STEPS = 2  
NUM_WORKERS = 8       
EPOCHS = 30
LEARNING_RATE = 1e-5   

# -----------------------------------------------------------------------------
# 3. Model and audio settings
# -----------------------------------------------------------------------------
NUM_CLASSES = 8
IMAGE_SIZE = 224

# [Audio Specs]
FPS = 30               
AUDIO_SAMPLE_RATE = 16000  

# Automatically compute the number of audio samples that match the video window.
MAX_AUDIO_LEN = int((SEQ_LEN / FPS) * AUDIO_SAMPLE_RATE)

# -----------------------------------------------------------------------------
# 4. Label and text mapping
# -----------------------------------------------------------------------------
EMOTION_MAP = {
    0: "Neutral",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
    7: "Other"
}

TEXT_TEMPLATE = "A face expressing {}"
EXPERIMENT_NAME = "cvpr_default"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BEST_CHECKPOINT_NAME = "best_model_vgcmf.pth"
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILENAME = "train.log"
SUMMARY_LOG_FILENAME = "metrics.txt"


def refresh_derived_config():
    global MAX_AUDIO_LEN
    MAX_AUDIO_LEN = int((SEQ_LEN / FPS) * AUDIO_SAMPLE_RATE)


def apply_case_config(case_path):
    with open(case_path, "r", encoding="utf-8") as f:
        case_cfg = yaml.safe_load(f)

    global TRAIN_ANNO_DIR, VAL_ANNO_DIR, IMAGE_BASE_DIRS, AUDIO_DIR
    global SEQ_LEN, STRIDE, IMAGE_SIZE, FPS, AUDIO_SAMPLE_RATE, NUM_CLASSES
    global BATCH_SIZE, NUM_WORKERS, GRAD_ACCUM_STEPS, EPOCHS, LEARNING_RATE
    global EXPERIMENT_NAME, CHECKPOINT_DIR, BEST_CHECKPOINT_NAME, LOG_DIR, LOG_FILENAME, SUMMARY_LOG_FILENAME

    data_cfg = case_cfg.get("data", {})
    loader_cfg = case_cfg.get("loader", {})
    optim_cfg = case_cfg.get("optim", {})
    checkpoint_cfg = case_cfg.get("checkpoint", {})
    log_cfg = case_cfg.get("log", {})

    EXPERIMENT_NAME = case_cfg.get("experiment_name", EXPERIMENT_NAME)
    TRAIN_ANNO_DIR = data_cfg.get("train_anno_dir", TRAIN_ANNO_DIR)
    VAL_ANNO_DIR = data_cfg.get("val_anno_dir", VAL_ANNO_DIR)
    IMAGE_BASE_DIRS = data_cfg.get("image_base_dirs", IMAGE_BASE_DIRS)
    AUDIO_DIR = data_cfg.get("audio_dir", AUDIO_DIR)
    SEQ_LEN = data_cfg.get("seq_len", SEQ_LEN)
    STRIDE = data_cfg.get("stride", STRIDE)
    IMAGE_SIZE = data_cfg.get("image_size", IMAGE_SIZE)
    FPS = data_cfg.get("fps", FPS)
    AUDIO_SAMPLE_RATE = data_cfg.get("audio_sample_rate", AUDIO_SAMPLE_RATE)
    NUM_CLASSES = data_cfg.get("num_classes", NUM_CLASSES)

    BATCH_SIZE = loader_cfg.get("batch_size", BATCH_SIZE)
    NUM_WORKERS = loader_cfg.get("num_workers", NUM_WORKERS)

    GRAD_ACCUM_STEPS = optim_cfg.get("grad_accum_steps", GRAD_ACCUM_STEPS)
    EPOCHS = optim_cfg.get("epochs", EPOCHS)
    LEARNING_RATE = optim_cfg.get("learning_rate", LEARNING_RATE)

    CHECKPOINT_DIR = checkpoint_cfg.get("dir", CHECKPOINT_DIR)
    BEST_CHECKPOINT_NAME = checkpoint_cfg.get("best_name", BEST_CHECKPOINT_NAME)
    LOG_DIR = log_cfg.get("dir", LOG_DIR)
    LOG_FILENAME = log_cfg.get("filename", LOG_FILENAME)
    SUMMARY_LOG_FILENAME = log_cfg.get("summary_filename", SUMMARY_LOG_FILENAME)

    refresh_derived_config()

# -----------------------------------------------------------------------------
# [Sanity Check] Configuration validation code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print(" [Config Sanity Check] Validating configuration...")
    print("="*60)
    
    paths_to_check = {
        "Train Anno": TRAIN_ANNO_DIR,
        "Val Anno": VAL_ANNO_DIR,
        "Audio Dir": AUDIO_DIR
    }
    
    for p in IMAGE_BASE_DIRS:
        paths_to_check[f"Image Dir ({os.path.basename(p)})"] = p

    all_exist = True
    for name, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"[OK] {name}")
        else:
            print(f"[MISSING] {name}: {path}")
            all_exist = False
    
    print("-" * 60)
    
    duration_sec = SEQ_LEN / FPS
    print(f"[INFO] Video Window: {SEQ_LEN} frames @ {FPS} fps = {duration_sec:.4f} sec")
    print(f"[INFO] Audio Required: {MAX_AUDIO_LEN} samples @ {AUDIO_SAMPLE_RATE} Hz")
    
    if MAX_AUDIO_LEN <= 0:
        print("[ERROR] Audio length is less than or equal to 0. Check SEQ_LEN or FPS.")
    else:
        print(f"[OK] The audio sample count ({MAX_AUDIO_LEN}) is logically valid.")

    print("-" * 60)
    if all_exist:
        print("All paths and settings look valid.")
    else:
        print("Warning: Some paths do not exist. Inference in `txt.py` may still work, but training will require fixes.")
    print("="*60)
