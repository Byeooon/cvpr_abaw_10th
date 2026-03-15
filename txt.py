import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torchaudio

import config
import model
from utils import set_seed

# Environment variable settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# [1] Custom dataset for the test set
# -----------------------------------------------------------------------------
class ABAWTestDataset(Dataset):
    def __init__(self, test_txt_path, test_data_dirs, audio_dir, seq_len=20, transform=None):
        self.test_data_dirs = test_data_dirs
        self.audio_dir = audio_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # 1. Load the list of test videos
        with open(test_txt_path, 'r') as f:
            video_names = [line.strip() for line in f if line.strip()]
            
        self.samples = []
        
        print("Building test data sequences...")
        for video_name in video_names:
            video_dir = self._get_video_dir(video_name)
            if not os.path.exists(video_dir):
                print(f"[Warning] Could not find video folder: {video_dir}")
                continue
                
            # 2. Sort image files (00001.jpg, 00002.jpg, etc.)
            frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
            if not frames:
                continue
                
            # 3. Build sliding-window sequences to predict each frame
            for i, target_frame in enumerate(frames):
                target_idx = i + 1 # 1-based index
                
                sequence = []
                for j in range(self.seq_len - 1, -1, -1):
                    idx = target_idx - j
                    if idx < 1:
                        idx = 1 # Pad with frame 1 if earlier frames are unavailable
                    sequence.append(f"{idx:05d}.jpg")
                    
                self.samples.append({
                    'video_name': video_name,
                    'target_frame_name': target_frame,
                    'sequence': sequence,
                    'start_frame_idx': max(1, target_idx - self.seq_len + 1)
                })
                
        print(f"Prepared {len(self.samples):,} test samples in total.")

    def _get_video_dir(self, video_name):
        for base_dir in self.test_data_dirs:
            candidate = os.path.join(base_dir, video_name)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.test_data_dirs[0], video_name)

    def _load_audio_chunk(self, video_name, start_frame):
        wav_path = os.path.join(self.audio_dir, f"{video_name}.wav")
        if not os.path.exists(wav_path):
            return torch.zeros(config.MAX_AUDIO_LEN)
            
        try:
            info = torchaudio.info(wav_path)
            orig_sr = info.sample_rate
            total_frames = info.num_frames
            
            start_sec = (start_frame - 1) / config.FPS 
            offset = int(start_sec * orig_sr)
            duration_sec = config.SEQ_LEN / config.FPS
            num_frames_to_load = int(duration_sec * orig_sr)
            
            if offset >= total_frames:
                return torch.zeros(config.MAX_AUDIO_LEN)
                
            waveform, sr = torchaudio.load(
                wav_path, frame_offset=offset, num_frames=num_frames_to_load, normalize=True
            )
            
            if sr != config.AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)
                
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            audio_tensor = waveform.squeeze(0)
            
            target_len = config.MAX_AUDIO_LEN
            current_len = audio_tensor.shape[0]
            
            if current_len < target_len:
                pad_size = target_len - current_len
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))
            elif current_len > target_len:
                audio_tensor = audio_tensor[:target_len]
                
            return audio_tensor
        except Exception:
            return torch.zeros(config.MAX_AUDIO_LEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample['video_name']
        target_frame_name = sample['target_frame_name']
        sequence = sample['sequence']
        start_frame = sample['start_frame_idx']
        
        images = []
        video_dir = self._get_video_dir(video_name)
        
        for frame_name in sequence:
            img_path = os.path.join(video_dir, frame_name)
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except:
                black_img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
                if self.transform:
                    black_img = self.transform(black_img)
                images.append(black_img)
                
        images_tensor = torch.stack(images)
        audio_tensor = self._load_audio_chunk(video_name, start_frame)
        
        image_location = f"{video_name}/{target_frame_name}"
        
        return images_tensor, audio_tensor, image_location

# -----------------------------------------------------------------------------
# [2] Result validation logic (checks contest submission format compliance)
# -----------------------------------------------------------------------------
def validate_submission_file(filepath):
    print("\n" + "=" * 60)
    print(" [Submission Format Validation] Checking the generated file format.")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print(f"[Error] Could not find file: {filepath}")
        return

    valid_classes = {'0', '1', '2', '3', '4', '5', '6', '7'}
    expected_header = "image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n"
    
        with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("[Error] The file is empty.")
        return

    # Header check
    if lines[0] != expected_header:
        print("[Error] The first-line header format does not match the contest specification.")
        print(f"  Expected: {expected_header.strip()}")
        print(f"  Actual: {lines[0].strip()}")
    else:
        print("[Pass] Header text matches.")

    # Content check
    format_errors = 0
    value_errors = 0
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) != 2:
            format_errors += 1
            continue
            
        img_loc, pred = parts
        if '/' not in img_loc or not img_loc.endswith('.jpg'):
            format_errors += 1
            
        if pred not in valid_classes:
            value_errors += 1

    total_predictions = len(lines) - 1
    print(f"[Pass] Verified a total of {total_predictions:,} predictions.")
    
    if format_errors == 0 and value_errors == 0:
        print("[Final Pass] All data formats fully comply with the specification.")
    else:
        print(f"[Warning] Location format errors: {format_errors}, value range errors (0-7): {value_errors}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# [3] Inference and file generation
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML case config")
    return parser.parse_args()


def generate_test_predictions():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    TEST_TXT_PATH = "/media/SSD/data/CVPR_workshop/test_release/names_of_videos_in_each_test_set/names_of_videos_in_each_test_set/Expression_Recognition_Challenge_test_set_release.txt"
    TEST_DATA_DIRS = [
        "/media/SSD/data/CVPR_workshop/cropped_aligned_image_test/batch1 (1)/cropped_aligned",
        "/media/SSD/data/CVPR_workshop/cropped_aligned_image_test/batch2/cropped_aligned_new_50_vids",
    ]

    print("Loading Test DataLoader...")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    test_dataset = ABAWTestDataset(
        test_txt_path=TEST_TXT_PATH,
        test_data_dirs=TEST_DATA_DIRS,
        audio_dir=config.AUDIO_DIR,
        seq_len=config.SEQ_LEN,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print("Loading VG-CMF Model...")
    model_instance = model.VGCMFEmotionModel(num_classes=config.NUM_CLASSES)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference with DataParallel.")
        model_instance = torch.nn.DataParallel(model_instance)
    model_instance = model_instance.to(device)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_CHECKPOINT_NAME)
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please run train.py first.")
        return

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    if isinstance(model_instance, torch.nn.DataParallel):
        model_instance.module.load_state_dict(model_state_dict)
    else:
        model_instance.load_state_dict(model_state_dict)
    model_instance.eval()
    
    output_file = os.path.join(os.getcwd(), "predictions.txt")
    print("-" * 60)
    print(f"Starting inference. Results will be written to '{output_file}'.")
    print("-" * 60)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n")
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Predicting", dynamic_ncols=True)
            
            for images, audio, image_locations in pbar:
                images = images.to(device, non_blocking=True)
                audio = audio.to(device, non_blocking=True)
                
                logits, _, _ = model_instance(
                    images,
                    audio,
                    input_ids=None,
                    attention_mask=None
                )
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                for loc, pred in zip(image_locations, preds):
                    f.write(f"{loc},{pred}\n")

                del images, audio, logits
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f" [Success] Inference complete. File saved to: {output_file}")
    validate_submission_file(output_file)

if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config.apply_case_config(args.config)
    generate_test_predictions()
