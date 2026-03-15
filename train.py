# train.py
import argparse
import os

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

import config
import data
import model
from utils import set_seed, get_class_weights, AverageMeter

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return distributed, rank, local_rank, world_size

def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def gather_tensor_across_ranks(tensor, world_size):
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML case config")
    return parser.parse_args()

def append_epoch_summary(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    summary_path = os.path.join(config.LOG_DIR, config.SUMMARY_LOG_FILENAME)

    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\ttrain_acc\ttrain_f1\tval_loss\tval_acc\tval_f1\n")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch}\t{train_loss:.4f}\t{train_acc:.2f}\t{train_f1:.2f}\t"
            f"{val_loss:.4f}\t{val_acc:.2f}\t{val_f1:.2f}\n"
        )

def train():
    distributed, rank, local_rank, world_size = setup_distributed()

    try:
        # 1. Fix the seed and configure the device
        set_seed(42 + rank)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
        else:
            device = torch.device("cpu")

        if is_main_process(rank):
            print(f"Device: {device}")
            print(f"Distributed: {distributed} | World Size: {world_size}")

        # 2. Prepare the data loaders
        if is_main_process(rank):
            print("Loading DataLoaders...")
        train_loader, val_loader, train_sampler, val_sampler = data.create_splits_and_loaders(
            distributed=distributed,
            rank=rank,
            world_size=world_size
        )
        
        # Compute class weights
        class_weights = get_class_weights(train_loader.dataset, num_classes=config.NUM_CLASSES)
        class_weights = class_weights.to(device)

        # 3. Initialize the model and tokenizer
        if is_main_process(rank):
            print("Loading Bi-directional VG-CMF Model and Tokenizer...")
        model_instance = model.VGCMFEmotionModel(num_classes=config.NUM_CLASSES).to(device)
        if distributed:
            model_instance = DDP(model_instance, device_ids=[local_rank], output_device=local_rank)
        model_module = model_instance.module if distributed else model_instance
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # 4. Configure loss functions and optimizer
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
        criterion_contrastive = nn.CrossEntropyLoss()

        trainable_params = [p for p in model_instance.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_CHECKPOINT_NAME)
        best_val_f1 = 0.0

        if is_main_process(rank):
            print("-" * 60)
            print(f"Experiment: {config.EXPERIMENT_NAME}")
            print(f"Summary Log: {os.path.join(config.LOG_DIR, config.SUMMARY_LOG_FILENAME)}")
            print("Start Training (Bi-directional VG-CMF)")
            print(f"Total Epochs: {config.EPOCHS}")
            print(f"Batch Size Per GPU: {config.BATCH_SIZE}")
            print(f"Sequence Length: {config.SEQ_LEN}")
            print(f"Stride: {config.STRIDE}")
            print(f"Gradient Accumulation Steps: {config.GRAD_ACCUM_STEPS}")
            print("-" * 60)

        for epoch in range(config.EPOCHS):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model_instance.train()
            
            losses = AverageMeter()
            top1 = AverageMeter()
            train_loss_sum = torch.zeros(1, device=device)
            train_sample_count = torch.zeros(1, device=device)
            train_correct_count = torch.zeros(1, device=device)
            all_train_preds = []
            all_train_labels = []
            
            optimizer.zero_grad()
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Train",
                dynamic_ncols=True,
                disable=not is_main_process(rank)
            )
            
            for step, (images, audio, texts, labels) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                audio = audio.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                text_inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)
                
                logits, visual_proj, text_features = model_instance(
                    images,
                    audio,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                )
                
                loss_cls = criterion_cls(logits, labels)
                
                if visual_proj is not None and text_features is not None:
                    logit_scale = model_module.clip.logit_scale.exp()
                    logits_per_video = logit_scale * visual_proj @ text_features.t()
                    logits_per_text = logits_per_video.t()
                    
                    contrastive_targets = torch.arange(len(labels), device=device)
                    loss_contrastive = (
                        criterion_contrastive(logits_per_video, contrastive_targets) +
                        criterion_contrastive(logits_per_text, contrastive_targets)
                    ) / 2
                    
                    loss = loss_cls + (0.1 * loss_contrastive)
                else:
                    loss = loss_cls

                loss = loss / config.GRAD_ACCUM_STEPS
                loss.backward()

                if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_size = images.size(0)
                preds = torch.argmax(logits, dim=1)
                
                acc = (preds == labels).float().mean().item() * 100.0

                losses.update(loss.item() * config.GRAD_ACCUM_STEPS, batch_size)
                top1.update(acc, batch_size)
                train_loss_sum += (loss.detach() * config.GRAD_ACCUM_STEPS) * batch_size
                train_sample_count += batch_size
                train_correct_count += (preds == labels).sum()
                all_train_preds.append(preds.detach())
                all_train_labels.append(labels.detach())
                
                if is_main_process(rank):
                    pbar.set_postfix({
                        'Loss': f"{losses.avg:.4f}",
                        'Acc': f"{top1.avg:.2f}%"
                    })

            if distributed:
                dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_sample_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_correct_count, op=dist.ReduceOp.SUM)

                local_train_pred_count = torch.tensor([sum(t.numel() for t in all_train_preds)], device=device, dtype=torch.long)
                train_pred_count_list = gather_tensor_across_ranks(local_train_pred_count, world_size)
                max_train_pred_count = max(int(t.item()) for t in train_pred_count_list)

                if all_train_preds:
                    local_train_preds = torch.cat(all_train_preds)
                    local_train_labels = torch.cat(all_train_labels)
                else:
                    local_train_preds = torch.empty(0, dtype=torch.long, device=device)
                    local_train_labels = torch.empty(0, dtype=torch.long, device=device)

                if local_train_preds.numel() < max_train_pred_count:
                    pad_size = max_train_pred_count - local_train_preds.numel()
                    local_train_preds = torch.cat([local_train_preds, torch.full((pad_size,), -1, dtype=torch.long, device=device)])
                    local_train_labels = torch.cat([local_train_labels, torch.full((pad_size,), -1, dtype=torch.long, device=device)])

                gathered_train_preds = gather_tensor_across_ranks(local_train_preds, world_size)
                gathered_train_labels = gather_tensor_across_ranks(local_train_labels, world_size)
            else:
                gathered_train_preds = [torch.cat(all_train_preds)] if all_train_preds else [torch.empty(0, dtype=torch.long, device=device)]
                gathered_train_labels = [torch.cat(all_train_labels)] if all_train_labels else [torch.empty(0, dtype=torch.long, device=device)]
                train_pred_count_list = [torch.tensor([gathered_train_preds[0].numel()], device=device, dtype=torch.long)]

            train_loss_avg = (train_loss_sum / train_sample_count.clamp(min=1)).item()
            train_acc_avg = (train_correct_count / train_sample_count.clamp(min=1)).item() * 100.0

            if is_main_process(rank):
                merged_train_preds = []
                merged_train_labels = []
                for preds_tensor, labels_tensor, count_tensor in zip(gathered_train_preds, gathered_train_labels, train_pred_count_list):
                    valid_count = int(count_tensor.item())
                    merged_train_preds.append(preds_tensor[:valid_count].cpu())
                    merged_train_labels.append(labels_tensor[:valid_count].cpu())

                merged_train_preds = torch.cat(merged_train_preds).numpy() if merged_train_preds else np.array([])
                merged_train_labels = torch.cat(merged_train_labels).numpy() if merged_train_labels else np.array([])
                train_f1_value = f1_score(merged_train_labels, merged_train_preds, average='macro', zero_division=0) * 100.0

            model_instance.eval()
            val_loss_sum = torch.zeros(1, device=device)
            val_sample_count = torch.zeros(1, device=device)
            val_correct_count = torch.zeros(1, device=device)
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Val",
                    dynamic_ncols=True,
                    disable=not is_main_process(rank)
                )
                
                for images, audio, texts, labels in val_pbar:
                    images = images.to(device, non_blocking=True)
                    audio = audio.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    text_inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)

                    logits, _, _ = model_instance(
                        images,
                        audio,
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs['attention_mask']
                    )
                    
                    loss = criterion_cls(logits, labels)
                    preds = torch.argmax(logits, dim=1)
                    batch_size = images.size(0)

                    val_loss_sum += loss.detach() * batch_size
                    val_sample_count += batch_size
                    val_correct_count += (preds == labels).sum()
                    all_val_preds.append(preds.detach())
                    all_val_labels.append(labels.detach())

                    if is_main_process(rank):
                        running_acc = (val_correct_count / val_sample_count.clamp(min=1)).item() * 100.0
                        running_loss = (val_loss_sum / val_sample_count.clamp(min=1)).item()
                        val_pbar.set_postfix({
                            'Val Loss': f"{running_loss:.4f}",
                            'Val Acc': f"{running_acc:.2f}%"
                        })

            if distributed:
                dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_sample_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_correct_count, op=dist.ReduceOp.SUM)

                local_pred_count = torch.tensor([sum(t.numel() for t in all_val_preds)], device=device, dtype=torch.long)
                pred_count_list = gather_tensor_across_ranks(local_pred_count, world_size)
                max_pred_count = max(int(t.item()) for t in pred_count_list)

                if all_val_preds:
                    local_preds = torch.cat(all_val_preds)
                    local_labels = torch.cat(all_val_labels)
                else:
                    local_preds = torch.empty(0, dtype=torch.long, device=device)
                    local_labels = torch.empty(0, dtype=torch.long, device=device)

                if local_preds.numel() < max_pred_count:
                    pad_size = max_pred_count - local_preds.numel()
                    local_preds = torch.cat([local_preds, torch.full((pad_size,), -1, dtype=torch.long, device=device)])
                    local_labels = torch.cat([local_labels, torch.full((pad_size,), -1, dtype=torch.long, device=device)])

                gathered_preds = gather_tensor_across_ranks(local_preds, world_size)
                gathered_labels = gather_tensor_across_ranks(local_labels, world_size)
            else:
                gathered_preds = [torch.cat(all_val_preds)] if all_val_preds else [torch.empty(0, dtype=torch.long, device=device)]
                gathered_labels = [torch.cat(all_val_labels)] if all_val_labels else [torch.empty(0, dtype=torch.long, device=device)]
                pred_count_list = [torch.tensor([gathered_preds[0].numel()], device=device, dtype=torch.long)]

            val_loss_avg = (val_loss_sum / val_sample_count.clamp(min=1)).item()
            val_acc_avg = (val_correct_count / val_sample_count.clamp(min=1)).item() * 100.0

            if is_main_process(rank):
                merged_preds = []
                merged_labels = []
                for preds_tensor, labels_tensor, count_tensor in zip(gathered_preds, gathered_labels, pred_count_list):
                    valid_count = int(count_tensor.item())
                    merged_preds.append(preds_tensor[:valid_count].cpu())
                    merged_labels.append(labels_tensor[:valid_count].cpu())

                merged_preds = torch.cat(merged_preds).numpy() if merged_preds else np.array([])
                merged_labels = torch.cat(merged_labels).numpy() if merged_labels else np.array([])
                val_f1_value = f1_score(merged_labels, merged_preds, average='macro', zero_division=0) * 100.0

            scheduler.step()

            if is_main_process(rank):
                print(f"\nEpoch [{epoch+1}/{config.EPOCHS}] Summary")
                print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc_avg:.2f}% | Train F1: {train_f1_value:.2f}")
                print(f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc_avg:.2f}% | Val F1: {val_f1_value:.2f} | Best Val F1: {best_val_f1:.2f}")
                print("-" * 60)
                append_epoch_summary(
                    epoch + 1,
                    train_loss_avg,
                    train_acc_avg,
                    train_f1_value,
                    val_loss_avg,
                    val_acc_avg,
                    val_f1_value,
                )

                if val_f1_value > best_val_f1:
                    best_val_f1 = val_f1_value
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_f1': best_val_f1,
                    }, checkpoint_path)
                    print(f"New Best Model Saved: {checkpoint_path}")

            if distributed:
                dist.barrier()
    finally:
        cleanup_distributed(distributed)

if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config.apply_case_config(args.config)
    train()
