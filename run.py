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
        # Lấy output của CrossAttention (thường là softmax results hoặc hidden states)
        # Chúng ta lọc lấy các lớp có độ phân giải latent 32x32 hoặc 64x64
        if isinstance(output, torch.Tensor) and output.shape[1] in [1024, 4096]:
            self.current_step_data.append(output.detach().cpu())

    def next_step(self):
        if len(self.current_step_data) > 0:
            self.step_attentions.append(self.current_step_data)
        self.current_step_data = []

def register_attention_hooks(model, store):
    # Xóa các hook cũ nếu có để tránh tràn bộ nhớ
    hooks = []
    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            hooks.append(module.register_forward_hook(store))
    return hooks

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
    model = prepare_model() # Sử dụng hàm prepare_model của bạn
    sampler = DDIMSampler(model)

    # Load và xử lý ảnh
    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    img_tensor = torch.from_numpy(np.array(raw_image)).float().div(127.5).sub(1.0).permute(2, 0, 1).unsqueeze(0).cuda()
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
    
    # Để có Attention Map rõ ràng, ta dùng prompt có từ khóa "eyebrows"
    prompt = "a man with eyebrows"
    c = model.get_learned_conditioning([prompt])

    # 1. DDIM Inversion (Tìm nhiễu gốc)
    print(f"[*] Đang thực hiện DDIM Inversion...")
    inverted_latent = ddim_inversion(model, sampler, init_latent, c, num_steps=steps)

    # 2. Đăng ký Hook để soi Attention
    store = AttentionStore()
    hooks = register_attention_hooks(model, store)

    # 3. Chạy quá trình Tái tạo (Sampling) thủ công
    # Lấy các tham số alpha từ sampler
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0, verbose=False)
    
    # SỬA LỖI TẠI ĐÂY: Chuyển sang numpy/list an toàn
    ddim_timesteps = sampler.ddim_timesteps.tolist()[::-1] # Đảo ngược list timestep
    ddim_alphas = sampler.ddim_alphas.cpu().numpy().tolist()[::-1] # Đảo ngược list alpha
    
    z = inverted_latent.clone()
    os.makedirs(os.path.join(BASE_DIR, "attn_maps"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reconstruction_steps"), exist_ok=True)

    print(f"[*] Đang tái tạo và trích xuất Attention Maps...")
    for i in range(len(ddim_timesteps)):
        t = torch.tensor([ddim_timesteps[i]], device=z.device)
        
        # Chạy UNet (Hook sẽ tự lưu data vào store)
        noise_pred = model.apply_model(z, t, c)
        
        # Tính bước DDIM tiếp theo
        alpha_cur = ddim_alphas[i]
        alpha_next = ddim_alphas[i+1] if i < len(ddim_alphas) - 1 else ddim_alphas[-1]
        
        z0_reconstructed = (z - np.sqrt(1 - alpha_cur) * noise_pred) / np.sqrt(alpha_cur)
        z = np.sqrt(alpha_next) * z0_reconstructed + np.sqrt(1 - alpha_next) * noise_pred
        
        # --- LƯU ATTENTION MAP ---
        if len(store.current_step_data) > 0:
            # Lấy layer attention cuối cùng của bước này
            attn = store.current_step_data[-1] 
            # Giả lập bản đồ nhiệt: Trung bình trên các đầu chú ý
            res = int(attn.shape[1]**0.5) # Thường là 32 hoặc 64
            # Gom thông tin không gian: (Batch, Pixels, Dim) -> (Res, Res)
            heatmap = attn[0].mean(dim=-1).reshape(res, res).numpy()
            
            plt.figure(figsize=(4,4))
            plt.imshow(heatmap, cmap='hot')
            plt.axis('off')
            plt.savefig(os.path.join(BASE_DIR, f"attn_maps/step_{i:03d}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        # Lưu ảnh tái tạo tại step này để so sánh
        if i % 5 == 0 or i == steps - 1:
            rec_img = model.decode_first_stage(z)
            rec_img = torch.clamp((rec_img + 1.0) / 2.0, min=0.0, max=1.0)
            rec_img = (rec_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
            Image.fromarray(rec_img).save(os.path.join(BASE_DIR, f"reconstruction_steps/step_{i:03d}.png"))
        
        store.next_step()

    # Gỡ hook sau khi xong để giải phóng GPU
    for h in hooks: h.remove()
    print(f"[*] Hoàn tất! Kết quả tại: {BASE_DIR}/attn_maps")

if __name__ == "__main__":
    execute_reconstruction(steps=50)