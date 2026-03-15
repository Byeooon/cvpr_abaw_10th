import os
import glob
import random
import torch
import torchaudio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import config

class ABAWExprSequenceDataset(Dataset):
    def __init__(self, txt_files, image_base_dirs, audio_dir, transform=None, dataset_name="Dataset"):
        self.txt_files = txt_files
        self.image_base_dirs = image_base_dirs
        self.audio_dir = audio_dir # [New] Audio path
        self.transform = transform
        self.dataset_name = dataset_name
        self.seq_len = config.SEQ_LEN
        
        # Updated sample structure: (video_name, start_frame_idx, sequence_list)
        self.samples = [] 
        
        # Cache audio resamplers for speed
        self.resamplers = {}

        self._build_dataset()

    def _get_video_dir(self, video_name):
        for base_dir in self.image_base_dirs:
            target_dir = os.path.join(base_dir, video_name)
            if os.path.exists(target_dir):
                return target_dir
        return None

    def _build_dataset(self):
        print(f"[{self.dataset_name}] Loading sequence data (Sequence Length: {self.seq_len})...")
        
        # [Added] Maximum number of missing frames allowed (e.g. 5 frames = about 0.16 sec)
        MAX_MISSING_TOLERANCE = 5 
        
        for txt_path in self.txt_files:
            video_name = os.path.splitext(os.path.basename(txt_path))[0]
            video_dir = self._get_video_dir(video_name)

            if not video_dir:
                continue

            existing_images = set([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            # Header handling: drop the first line if it contains labels such as 'neutral'
            if len(lines) > 0 and 'neutral' in lines[0].lower():
                labels = lines[1:]
            else:
                labels = lines

            current_sequence = []
            
            # [Added] State tracking variables for interpolation
            last_valid_frame_data = None
            missing_count = 0

            for idx, label_str in enumerate(labels):
                try:
                    label = int(label_str.strip())
                except ValueError:
                    # If it is not numeric, force it to an invalid label (-1)
                    label = -1 
                
                # Frame indices start from 1 (00001.jpg)
                frame_idx = idx + 1
                img_filename = f"{frame_idx:05d}.jpg"
                
                # Check whether the current frame is valid
                is_valid = (label != -1) and (img_filename in existing_images)

                if is_valid:
                    img_path = os.path.join(video_dir, img_filename)
                    # Add a valid frame and update the cached state
                    last_valid_frame_data = (img_path, label)
                    current_sequence.append((img_path, label, frame_idx))
                    missing_count = 0  # Reset missing-frame count
                    
                else:
                    # [Updated logic] If the frame is invalid, copy the last valid frame
                    if last_valid_frame_data is not None and missing_count < MAX_MISSING_TOLERANCE:
                        last_img_path, last_label = last_valid_frame_data
                        # Reuse the previous image but keep the current frame index for audio sync
                        current_sequence.append((last_img_path, last_label, frame_idx))
                        missing_count += 1
                    else:
                        # If tolerance is exceeded or frames are missing from the start, reset the sequence
                        current_sequence = []
                        last_valid_frame_data = None
                        missing_count = 0
                
                # Save a sample once the target sequence length is reached
                if len(current_sequence) == self.seq_len:
                    start_frame = current_sequence[0][2]
                    self.samples.append({
                        'video_name': video_name,
                        'sequence': current_sequence.copy(),
                        'start_frame': start_frame
                    })
                    
                    # [Key update] Apply sliding-window stride so windows overlap partially
                    # Use the logic below instead of the old `current_sequence.pop(0)`
                    window_stride = config.STRIDE
                    current_sequence = current_sequence[window_stride:]
                    
        print(f"-> {self.dataset_name} ready: total {len(self.samples):,} sequences")

    def _load_audio_chunk(self, video_name, start_frame):
        """
        Slice the audio chunk that corresponds to the video's `start_frame`.
        """
        wav_path = os.path.join(self.audio_dir, f"{video_name}.wav")
        
        # 1. If the file does not exist, fill with zeros
        if not os.path.exists(wav_path):
            return torch.zeros(config.MAX_AUDIO_LEN)
            
        try:
            # 2. Check metadata
            info = torchaudio.info(wav_path)
            orig_sr = info.sample_rate
            total_frames = info.num_frames
            
            # 3. Compute the read offset (watch the 0-based index conversion)
            # `start_frame` begins at 1, so subtract 1 here
            start_sec = (start_frame - 1) / config.FPS 
            offset = int(start_sec * orig_sr)
            
            # Read length matching the video window duration
            duration_sec = config.SEQ_LEN / config.FPS
            num_frames_to_load = int(duration_sec * orig_sr)
            
            # Range check
            if offset >= total_frames:
                return torch.zeros(config.MAX_AUDIO_LEN)
                
            # 4. Partial loading (load chunk)
            waveform, sr = torchaudio.load(
                wav_path,
                frame_offset=offset,
                num_frames=num_frames_to_load,
                normalize=True
            )
            
            # 5. Resample to 16 kHz
            if sr != config.AUDIO_SAMPLE_RATE:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, config.AUDIO_SAMPLE_RATE)
                waveform = self.resamplers[sr](waveform)
            
            # 6. Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            audio_tensor = waveform.squeeze(0)

            # 7. Fix the length (padding / truncation)
            target_len = config.MAX_AUDIO_LEN
            current_len = audio_tensor.shape[0]
            
            if current_len < target_len:
                pad_size = target_len - current_len
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
            elif current_len > target_len:
                audio_tensor = audio_tensor[:target_len]
                
            return audio_tensor

        except Exception as e:
            return torch.zeros(config.MAX_AUDIO_LEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sequence = sample['sequence']
        video_name = sample['video_name']
        start_frame = sample['start_frame']
        
        # --- [1] Load image sequence ---
        images = []
        target_label = sequence[-1][1] # The last-frame label is the target
        
        for img_path, _, _ in sequence:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except:
                # Fallback for corrupted images (black image)
                black_img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
                if self.transform:
                    black_img = self.transform(black_img)
                images.append(black_img)
                
        images_tensor = torch.stack(images) # (SEQ_LEN, C, H, W)
        
        # --- [2] Load audio chunk (New) ---
        audio_tensor = self._load_audio_chunk(video_name, start_frame) # (MAX_AUDIO_LEN,)
        
        # --- [3] Text prompt ---
        emotion_str = config.EMOTION_MAP[target_label]
        text_prompt = config.TEXT_TEMPLATE.format(emotion_str)
            
        return images_tensor, audio_tensor, text_prompt, target_label

def create_splits_and_loaders(distributed=False, rank=0, world_size=1):
    # 1. Training file list
    train_files = glob.glob(os.path.join(config.TRAIN_ANNO_DIR, "*.txt"))
    
    # 2. Validation file list (using config paths)
    val_files = glob.glob(os.path.join(config.VAL_ANNO_DIR, "*.txt"))
    
    # Keep the original preprocessing pipeline (CLIP mean/std)
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Create datasets
    train_dataset = ABAWExprSequenceDataset(
        train_files, 
        config.IMAGE_BASE_DIRS, 
        config.AUDIO_DIR, 
        transform=transform, 
        dataset_name="Train"
    )
    
    val_dataset = ABAWExprSequenceDataset(
        val_files, 
        config.IMAGE_BASE_DIRS, 
        config.AUDIO_DIR, 
        transform=transform, 
        dataset_name="Validation"
    )
    
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler

# -----------------------------------------------------------------------------
# [Sanity Check] Data loading test code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print(" [Data Sanity Check] Testing data loading...")
    print("="*60)
    
    try:
        # Initialize the dataset with just one file for testing
        test_files = glob.glob(os.path.join(config.TRAIN_ANNO_DIR, "*.txt"))[:1]
        
        dataset = ABAWExprSequenceDataset(
            test_files, 
            config.IMAGE_BASE_DIRS, 
            config.AUDIO_DIR, 
            transform=transforms.ToTensor(), # Simple transform for testing
            dataset_name="Test-Single"
        )
        
        if len(dataset) == 0:
            print("[ERROR] The dataset is empty. Check the paths.")
        else:
            print(f"[OK] Dataset loaded successfully. Number of samples: {len(dataset)}")
            
            # Load the first sample and inspect its shapes
            print("-" * 60)
            print(" Loading the first sample...")
            images_tensor, audio_tensor, text, label = dataset[0]
            
            print(f" [Visual] Shape: {images_tensor.shape} (Expect: {config.SEQ_LEN}, 3, 224, 224)")
            print(f" [Audio]  Shape: {audio_tensor.shape} (Expect: {config.MAX_AUDIO_LEN})")
            print(f" [Text]   Prompt: \"{text}\"")
            print(f" [Label]  Class: {label}")
            
            # Check the audio data status
            if torch.all(audio_tensor == 0):
                print(" [WARNING] The audio data is all zeros. (Missing file or silence)")
            else:
                print(f" [OK] Audio data is present (Mean: {audio_tensor.mean():.4f}, Std: {audio_tensor.std():.4f})")
                
            print("-" * 60)
            print(" [Success] The data pipeline is working correctly.")

    except Exception as e:
        print(f" [ERROR] An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)
