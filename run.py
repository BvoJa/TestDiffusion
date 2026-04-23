import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import requests
from tqdm import tqdm
import torch.nn.functional as F

# --- THIẾT LẬP ĐƯỜNG DẪN ---
BASE_DIR = "/kaggle/working/TestDiffusion"
IMAGE_PATH = "/kaggle/input/datasets/lhgbao/faceee/face.png" 

# Cấu hình Model và Checkpoint cho v1.5
# Sử dụng bản pruned-emaonly để tối ưu cho việc tái tạo ảnh
CKPT_URL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
CKPT_PATH = os.path.join(BASE_DIR, "v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(BASE_DIR, "configs/stable-diffusion/v1-inference.yaml")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# --- [PHẦN MỚI] 1: Lớp ghi lại Attention Map ---
class AttentionStore:
    def __init__(self):
        self.step_attentions = []
        self.current_step_data = []

    def __call__(self, module, input, output):
        # Ma trận chú ý thường có dạng: (batch * heads, query_length, context_length)
        # Với ảnh 512x512, query_length có thể là 4096 (64x64), 1024 (32x32), v.v.
        # Chúng ta lấy trung bình của các heads để dễ quan sát
        if output.shape[1] <= 4096: # Chỉ lấy các layer có độ phân giải đủ lớn (64x64 hoặc 32x32)
            self.current_step_data.append(output.cpu())

    def next_step(self):
        if len(self.current_step_data) > 0:
            # Lấy trung bình tất cả các layer attention trong 1 bước
            all_layers = torch.cat([attn.flatten(1) for attn in self.current_step_data], dim=1)
            self.step_attentions.append(self.current_step_data)
            self.current_step_data = []

def register_attention_hooks(model, store):
    def hook_fn(module, input, output):
        # Trong kiến trúc LDM, CrossAttention tính toán attention và trả về output
        # Chúng ta cần capture ma trận softmax bên trong, nhưng để đơn giản, 
        # ta sẽ lấy output của lớp attention (đã được nhân với giá trị) 
        # hoặc can thiệp sâu hơn. Ở đây ta giả định ghi lại để xem vùng kích hoạt.
        store(module, input, output)

    # Lặp qua các module để tìm CrossAttention
    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            # Chỉ hook vào các lớp Cross-Attention (liên quan đến Prompt)
            module.register_forward_hook(store)

def prepare_model():
    # 1. Tải checkpoint v1.5
    if not os.path.exists(CKPT_PATH):
        print(f"[*] Đang tải checkpoint SD v1.5 (4.27GB)... Vui lòng đợi.")
        response = requests.get(CKPT_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading v1.5")
        
        with open(CKPT_PATH, 'wb') as file:
            for data in response.iter_content(chunk_size=1024*1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("\n[*] Tải xong!")
    else:
        print(f"[*] Checkpoint v1.5 đã tồn tại tại {CKPT_PATH}.")
    
    # 2. Load Model
    config = OmegaConf.load(CONFIG_PATH)
    print(f"[*] Đang nạp model v1.5 vào GPU...")
    # Lưu ý: PyTorch 1.11 (Conda của bạn) không cần weights_only=False
    pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

@torch.no_grad()
def ddim_inversion(model, sampler, latent, cond, num_steps=50):
    sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=0, verbose=False)
    alphas = sampler.ddim_alphas
    z = latent.clone()
    for i in range(num_steps):
        model_t = torch.tensor([sampler.ddim_timesteps[i]], device=latent.device)
        noise_pred = model.apply_model(z, model_t, cond) 
        
        alpha_cur = alphas[i]
        alpha_next = alphas[i+1] if i < num_steps - 1 else alphas[-1]
        z0_reconstructed = (z - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
        z = alpha_next.sqrt() * z0_reconstructed + (1 - alpha_next).sqrt() * noise_pred
    return z

@torch.no_grad()
def execute_reconstruction(steps=50):
    # (Giữ nguyên phần prepare_model và load ảnh của bạn)
    from ldm.models.diffusion.ddim import DDIMSampler
    model = prepare_model()
    sampler = DDIMSampler(model)
    
    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    img_tensor = torch.from_numpy(np.array(raw_image)).float().div(127.5).sub(1.0).permute(2, 0, 1).unsqueeze(0).cuda()
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
    
    # Prompt: Nên dùng prompt có nghĩa để Attention Map rõ ràng hơn
    prompt = "a man with eyebrows" 
    c = model.get_learned_conditioning([prompt])

    # 1. DDIM Inversion
    print(f"[*] Inverting...")
    inverted_latent = ddim_inversion(model, sampler, init_latent, c, num_steps=steps)

    # 2. Thiết lập Hook
    attn_store = AttentionStore()
    register_attention_hooks(model, attn_store)

    # 3. Vòng lặp tái tạo thủ công (thay cho sampler.sample)
    # Để lưu được map ở mỗi bước, ta chạy thủ công từng bước DDIM
    os.makedirs(os.path.join(BASE_DIR, "attn_maps"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reconstruction_steps"), exist_ok=True)
    
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0, verbose=False)
    timesteps = np.flip(sampler.ddim_timesteps) # Chạy từ nhiễu về ảnh rõ
    alphas = np.flip(sampler.ddim_alphas)
    
    z = inverted_latent.clone()
    
    print(f"[*] Bắt đầu tái tạo và lưu Attention Map...")
    for i, t in enumerate(tqdm(timesteps)):
        ts = torch.tensor([t], device=z.device)
        
        # Bước UNet: Hook sẽ tự động lưu dữ liệu vào attn_store ở đây
        noise_pred = model.apply_model(z, ts, c)
        
        # Tính bước tiếp theo (DDIM Step)
        alpha_cur = alphas[i]
        alpha_prev = alphas[i+1] if i < steps - 1 else alphas[-1]
        z0_reconstructed = (z - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
        z = alpha_prev.sqrt() * z0_reconstructed + (1 - alpha_prev).sqrt() * noise_pred
        
        # --- LƯU ẢNH TRUNG GIAN ---
        rec_step = model.decode_first_stage(z)
        rec_step = torch.clamp((rec_step + 1.0) / 2.0, min=0.0, max=1.0)
        rec_img = (rec_step.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        Image.fromarray(rec_img).save(os.path.join(BASE_DIR, f"reconstruction_steps/step_{i:03d}.png"))

        # --- LƯU ATTENTION MAP ---
        if len(attn_store.current_step_data) > 0:
            # Lấy 1 map tiêu biểu (ví dụ layer cuối của UNet, thường là layer 0 hoặc -1)
            # Reshape về dạng 2D. Giả sử map là 16x16 hoặc 32x32 hoặc 64x64
            last_attn = attn_store.current_step_data[-1] # Lấy layer cuối cùng
            
            # Tính trung bình trên các heads và các tokens để ra bản đồ nhiệt không gian
            # (Hoặc bạn có thể chọn token index cụ thể cho "eyebrows")
            spatial_map = last_attn.mean(dim=0).mean(dim=-1) 
            res = int(spatial_map.shape[0]**0.5)
            spatial_map = spatial_map.reshape(res, res).numpy()
            
            # Vẽ và lưu Heatmap
            plt.figure(figsize=(5,5))
            plt.imshow(spatial_map, cmap='jet')
            plt.title(f"Attention Map Step {i}")
            plt.axis('off')
            plt.savefig(os.path.join(BASE_DIR, f"attn_maps/attn_{i:03d}.png"))
            plt.close()
            
        attn_store.next_step() # Reset cho bước sau

    print(f"[*] Xong! Kiểm tra thư mục 'attn_maps' và 'reconstruction_steps'")

if __name__ == "__main__":
    execute_reconstruction(steps=50)