import os

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Memory fragmentation mitigation (helps with OOM issues)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import CLIPTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np

import config
import data
import model
from utils import set_seed

def evaluate():
    # 1. Fix the seed and configure the device
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # [Updated OOM handling]
    # The 90-frame setup can force the default batch size to be small (e.g. 8).
    # For evaluation, keeping the config batch size or reducing it by half is usually safe.
    print(f"Original Config Batch Size: {config.BATCH_SIZE}")
    # Evaluation does not store training graphs, so using the training batch size is typically safe.
    # If OOM still occurs, change this to `config.BATCH_SIZE // 2`.
    eval_batch_size = config.BATCH_SIZE 
    config.BATCH_SIZE = eval_batch_size
    print(f"Evaluation Batch Size: {config.BATCH_SIZE} (matched to the config for the 90-frame setup)")

    # 2. Prepare the data loader
    print("Loading DataLoaders...")
    _, val_loader, _ = data.create_splits_and_loaders()

    # 3. Initialize the model and tokenizer
    # The internal model architecture changed to Bi-directional VG-CMF,
    # but the input/output interface is unchanged, so it can be used as-is.
    print("Loading Bi-directional VG-CMF Model and Tokenizer...")
    model_instance = model.VGCMFEmotionModel(num_classes=config.NUM_CLASSES).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 4. Load the best saved checkpoint
    # [Updated] Apply the new absolute path instead of the previous relative path
    checkpoint_path = os.path.join("checkpoints", "best_model_vgcmf.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please run train.py first.")
        return

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    
    saved_epoch = checkpoint.get('epoch', 0)
    best_f1 = checkpoint.get('best_val_f1', 0.0)
    print(f"Load complete (Saved at Epoch {saved_epoch + 1}, Best Val F1: {best_f1:.4f})")

    # 5. Start the evaluation loop
    model_instance.eval()
    
    all_preds = []
    all_labels = []
    
    print("-" * 60)
    print("Starting evaluation on Validation/Test set...")
    print("-" * 60)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", dynamic_ncols=True)
        
        for images, audio, texts, labels in pbar:
            images = images.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Tokenize text
            text_inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)

            # Forward pass (Bi-directional model inference)
            logits, _, _ = model_instance(
                images, 
                audio, 
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Free memory to reduce OOM risk
            del images, audio, labels, logits
            torch.cuda.empty_cache()

    # 6. Compute evaluation metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    correct = (all_preds == all_labels).sum()
    accuracy = (correct / len(all_labels)) * 100.0

    print("\n" + "=" * 60)
    print(" Final Evaluation Results")
    print("=" * 60)
    print(f"Total Samples: {len(all_labels):,}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("-" * 60)

    target_names = [config.EMOTION_MAP[i] for i in range(config.NUM_CLASSES)]
    
    print("\n[Detailed Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, digits=4))
    
    print("\n[Confusion Matrix]")
    print(confusion_matrix(all_labels, all_preds))
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
