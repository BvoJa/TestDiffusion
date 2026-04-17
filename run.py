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

def execute_reconstruction(steps=50):
    model = prepare_model()
    sampler = DDIMSampler(model)

    if not os.path.exists(IMAGE_PATH):
        print(f"LỖI: Không tìm thấy ảnh tại {IMAGE_PATH}.")
        return

    # Resize về 512x512 để khớp với kiến trúc v1.5
    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
    img_array = np.array(raw_image)
    img_tensor = torch.from_numpy(img_array).float().div(127.5).sub(1.0).permute(2, 0, 1).unsqueeze(0).cuda()

    init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
    c = model.get_learned_conditioning([""])

    print(f"[*] Bắt đầu quy trình Inversion ({steps} steps)...")
    inverted_latent = ddim_inversion(model, sampler, init_latent, c, num_steps=steps)

    print(f"[*] Bắt đầu quy trình Tái tạo...")
    samples, _ = sampler.sample(
        S=steps, 
        batch_size=1, 
        shape=init_latent.shape[1:], 
        conditioning=c, 
        eta=0.0, 
        x_T=inverted_latent
    )

    rec_latent = model.decode_first_stage(samples)
    rec_latent = torch.clamp((rec_latent + 1.0) / 2.0, min=0.0, max=1.0)
    rec_image = (rec_latent.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    # Plot ngầm
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(rec_image)
    axes[1].set_title(f"v1.5 Reconstruction ({steps} steps)")
    axes[1].axis('off')
    plt.tight_layout()
    
    # Lưu kết quả
    result_name = "face_reconstructed.png"
    Image.fromarray(rec_image).save(os.path.join(BASE_DIR, result_name))
    print(f"[*] Hoàn tất! Kết quả: {BASE_DIR}/{result_name}")

if __name__ == "__main__":
    execute_reconstruction(steps=100)