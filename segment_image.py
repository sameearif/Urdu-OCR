
from utils import *
from glob import glob
from tqdm import tqdm

model = load_sr_model("models/super-resolution/swinir.pth", device="mps")

img = prepare_image("305.png", "mps")
with torch.no_grad():
    out = model(img)

    to_pil = transforms.ToPILImage()
    sr = to_pil(out.clamp(0,1).cpu().squeeze(0))
    sr.save(f"1.png")