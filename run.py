import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # THÊM DÒNG NÀY VÀO TRƯỚC LÚC IMPORT PYPLOT
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import requests
from tqdm import tqdm

# --- THIẾT LẬP ĐƯỜNG DẪN ---
BASE_DIR = "/kaggle/working/TestDiffusion"
# Đường dẫn ảnh bạn đã cung cấp
IMAGE_PATH = "/kaggle/input/datasets/lhgbao/portrait/trump.jpg" 

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# Cấu hình Model và Checkpoint
CKPT_URL = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
CKPT_PATH = os.path.join(BASE_DIR, "sd-v1-4.ckpt")
CONFIG_PATH = os.path.join(BASE_DIR, "configs/stable-diffusion/v1-inference.yaml")

def prepare_model():
    # 1. Tải checkpoint bằng Python Requests + TQDM
    if not os.path.exists(CKPT_PATH):
        print(f"[*] Đang tải checkpoint 4GB... Vui lòng đợi.")
        
        # Mở kết nối tải file
        response = requests.get(CKPT_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Tạo thanh tiến trình đẹp mắt
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        
        # Ghi file theo từng cục (chunk)
        with open(CKPT_PATH, 'wb') as file:
            for data in response.iter_content(chunk_size=1024*1024): # Tải từng 1MB
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        print("\n[*] Tải xong!")
    else:
        print("[*] Checkpoint đã tồn tại, bỏ qua bước tải.")
    
    # 2. Load Model (phần này giữ nguyên như cũ)
    config = OmegaConf.load(CONFIG_PATH)
    print(f"[*] Đang nạp model vào GPU...")
    pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

@torch.no_grad()
def ddim_inversion(model, sampler, latent, cond, num_steps=50): # Đã thêm cond
    sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=0, verbose=False)
    alphas = sampler.ddim_alphas
    z = latent.clone()
    for i in range(num_steps):
        model_t = torch.tensor([sampler.ddim_timesteps[i]], device=latent.device)
        
        # Thay None bằng cond ở đây
        noise_pred = model.apply_model(z, model_t, cond) 
        
        alpha_cur = alphas[i]
        alpha_next = alphas[i+1] if i < num_steps - 1 else alphas[-1]
        z0_reconstructed = (z - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
        z = alpha_next.sqrt() * z0_reconstructed + (1 - alpha_next).sqrt() * noise_pred
    return z

def execute_reconstruction(steps=50):
    model = prepare_model()
    sampler = DDIMSampler(model)

    # 1. Xử lý ảnh gốc từ đường dẫn của bạn
    if not os.path.exists(IMAGE_PATH):
        print(f"LỖI: Không tìm thấy ảnh tại {IMAGE_PATH}. Hãy kiểm tra lại dataset đã được add vào Kaggle chưa.")
        return

    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    img_array = np.array(raw_image)
    img_tensor = torch.from_numpy(img_array).float().div(127.5).sub(1.0).permute(2, 0, 1).unsqueeze(0).cuda()

    # 2. Encode sang Latent Space
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))

    c = model.get_learned_conditioning([""])

    # 3. DDIM Inversion (Tìm nhiễu xác định)
    print(f"[*] Bắt đầu quy trình Inversion...")
    inverted_latent = ddim_inversion(model, sampler, init_latent, c, num_steps=steps)

    # 4. DDIM Reconstruction (Tái tạo từ nhiễu đó)
    print(f"[*] Bắt đầu quy trình Tái tạo...")
    samples, _ = sampler.sample(
        S=steps, 
        batch_size=1, 
        shape=init_latent.shape[1:], 
        conditioning=c, 
        eta=0.0, 
        x_T=inverted_latent
    )

    # 5. Giải mã (Decode) về ảnh thường
    rec_latent = model.decode_first_stage(samples)
    rec_latent = torch.clamp((rec_latent + 1.0) / 2.0, min=0.0, max=1.0)
    rec_image = (rec_latent.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    # 6. Plot so sánh ảnh gốc và kết quả
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_array)
    axes[0].set_title("Ảnh Gốc (Original Portrait)")
    axes[0].axis('off')

    axes[1].imshow(rec_image)
    axes[1].set_title(f"Ảnh Tái tạo (DDIM Reconstruction - {steps} steps)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Lưu kết quả
    Image.fromarray(rec_image).save(os.path.join(BASE_DIR, "trump_reconstructed.png"))
    print(f"[*] Hoàn tất! Ảnh đã được lưu tại {BASE_DIR}/trump_reconstructed.png")

# Chạy script
if __name__ == "__main__":
    execute_reconstruction(steps=50)