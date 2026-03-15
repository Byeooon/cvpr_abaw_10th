# utils.py
import torch
import numpy as np
import random
from collections import Counter

def set_seed(seed=42):
    """
    Fix the random seed to ensure experiment reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(dataset, num_classes=8):
    """
    Analyze the label distribution and compute inverse-frequency loss weights.
    """
    print("-" * 60)
    print("Analyzing label distribution and calculating weights...")
    
    # In the updated `data.py` structure, `sample['sequence'][-1][1]` is the last frame label.
    labels = [sample['sequence'][-1][1] for sample in dataset.samples]
    
    total_samples = len(labels)
    label_counts = Counter(labels)
    counts = np.array([label_counts.get(i, 0) for i in range(num_classes)])
    
    print("-" * 60)
    print("Label Distribution:")
    for i in range(num_classes):
        percentage = (counts[i] / total_samples) * 100 if total_samples > 0 else 0
        print(f"Class {i}: {counts[i]:>7} samples ({percentage:.2f}%)")
        
    # Weights = total_samples / (num_classes * counts)
    weights = total_samples / (num_classes * (counts + 1e-5))
    
    print("-" * 60)
    print("Calculated Loss Weights:")
    for i in range(num_classes):
        print(f"Class {i} Weight: {weights[i]:.4f}")
    print("-" * 60)
    
    return torch.FloatTensor(weights)

class AverageMeter(object):
    """
    Tracks average metric values during training and validation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
