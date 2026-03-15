# model.py
import torch
import torch.nn as nn
from transformers import CLIPModel, Wav2Vec2Model
import config

# -----------------------------------------------------------------------------
# [1] TCN Modules
# -----------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -----------------------------------------------------------------------------
# [2] New Model: Bi-directional VG-CMF
# -----------------------------------------------------------------------------
class VGCMFEmotionModel(nn.Module):
    def __init__(self, num_classes=8, clip_model_name="openai/clip-vit-base-patch32", wav2vec_model_name="facebook/wav2vec2-base-960h"):
        super(VGCMFEmotionModel, self).__init__()
        
        # --- 1. Visual Backbone (CLIP - Frozen) ---
        print(f"[Model] Loading CLIP: {clip_model_name} ...")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        embed_dim = self.clip.config.projection_dim # 512
        
        # --- 2. Audio Backbone (Wav2Vec 2.0 - Frozen) ---
        print(f"[Model] Loading Wav2Vec2: {wav2vec_model_name} ...")
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        for param in self.wav2vec.parameters():
            param.requires_grad = False
            
        audio_dim = self.wav2vec.config.hidden_size # 768
        
        # --- 3. Encoders & Adapters ---
        # [Visual TCN] Increase depth to cover 90 frames (3 layers -> 6 layers)
        tcn_channels = [embed_dim] * 6 
        self.visual_tcn = TemporalConvNet(num_inputs=embed_dim, num_channels=tcn_channels, kernel_size=3, dropout=0.4)
        
        # [Audio Adapter]
        self.audio_adapter = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # --- 4. Bi-directional Fusion Module ---
        # [V2A] Video Query, Audio Key/Value
        self.v2a_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.v2a_norm = nn.LayerNorm(embed_dim)
        
        # [A2V] Audio Query, Video Key/Value
        self.a2v_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.a2v_norm = nn.LayerNorm(embed_dim)
        
        # --- 5. Classifier ---
        # Input: Concat(Pooled Video_enchanced, Pooled Audio_enhanced) -> 1024
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased to improve overfitting resistance
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Projection for text contrastive learning
        self.video_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, images, audio, input_ids=None, attention_mask=None):
        B, S, C, H, W = images.shape
        
        # ==========================
        # 1. Visual Stream
        # ==========================
        images = images.view(B * S, C, H, W)
        with torch.no_grad():
            # In transformers v5, `get_image_features` may return a different type,
            # so call the vision backbone and projection directly.
            vision_backbone = self.clip.vision_model(pixel_values=images)
            vision_outputs = self.clip.visual_projection(vision_backbone.pooler_output) # (B*S, 512)
        
        vision_seq = vision_outputs.view(B, S, -1).permute(0, 2, 1)
        tcn_out = self.visual_tcn(vision_seq) 
        video_feat = tcn_out.permute(0, 2, 1) # (B, S, 512)

        # ==========================
        # 2. Audio Stream
        # ==========================
        with torch.no_grad():
            audio_out = self.wav2vec(audio).last_hidden_state # (B, T_audio, 768)
            
        audio_feat = self.audio_adapter(audio_out) # (B, T_audio, 512)

        # ==========================
        # 3. Bi-directional Fusion
        # ==========================
        # V2A: Video Query
        v2a_out, _ = self.v2a_attention(query=video_feat, key=audio_feat, value=audio_feat)
        fused_video = self.v2a_norm(video_feat + v2a_out) 
        
        # A2V: Audio Query
        a2v_out, _ = self.a2v_attention(query=audio_feat, key=video_feat, value=video_feat)
        fused_audio = self.a2v_norm(audio_feat + a2v_out)

        # ==========================
        # 4. Pooling & Classification
        # ==========================
        pooled_video = torch.mean(fused_video, dim=1) # (B, 512)
        pooled_audio = torch.mean(fused_audio, dim=1) # (B, 512)
        
        final_feat = torch.cat([pooled_video, pooled_audio], dim=-1) # (B, 1024)
        
        logits = self.classifier(final_feat)
        
        # ==========================
        # 5. Text Features (Optional)
        # ==========================
        if input_ids is not None:
            visual_proj = self.video_projection(pooled_video)
            visual_proj = visual_proj / visual_proj.norm(p=2, dim=-1, keepdim=True)
            
            with torch.no_grad():
                text_backbone = self.clip.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_out = self.clip.text_projection(text_backbone.pooler_output)
                text_features = text_out / text_out.norm(p=2, dim=-1, keepdim=True)
                
            return logits, visual_proj, text_features

        return logits, None, None

# -----------------------------------------------------------------------------
# [Sanity Check] Model structure and forward-pass validation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print(" [Model Sanity Check] Starting Bi-directional VG-CMF model validation...")
    print("="*60)

    try:
        print("1. Initializing model...")
        model = VGCMFEmotionModel(num_classes=config.NUM_CLASSES)
        model.eval()
        print("[OK] Model initialization complete.")

        print("-" * 60)
        print("2. Creating dummy data...")
        batch_size = 2
        seq_len = config.SEQ_LEN
        img_size = config.IMAGE_SIZE
        audio_len = config.MAX_AUDIO_LEN

        dummy_video = torch.randn(batch_size, seq_len, 3, img_size, img_size)
        dummy_audio = torch.randn(batch_size, audio_len)

        print(f" [Input] Video: {dummy_video.shape}")
        print(f" [Input] Audio: {dummy_audio.shape}")

        print("-" * 60)
        print("3. Running forward pass...")
        with torch.no_grad():
            logits, _, _ = model(dummy_video, dummy_audio)

        print("-" * 60)
        print(f" [Output] Logits Shape: {logits.shape}")
        
        expected_shape = (batch_size, config.NUM_CLASSES)
        if logits.shape == expected_shape:
            print(f"[OK] Output shape matches the expected shape: {expected_shape}")
            print("Bi-directional VG-CMF model validation succeeded.")
        else:
            print(f"[ERROR] Output shape mismatch. Expected: {expected_shape}, Actual: {logits.shape}")

    except Exception as e:
        print(f"[ERROR] An error occurred during model validation: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)
