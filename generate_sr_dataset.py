import argparse
import os
import io
import random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Path to raw input images")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save HR images")
parser.add_argument("--val_split", type=float, required=True, help="Validation Split in Ratio")
args = parser.parse_args()

def create_dir():
    os.makedirs(os.path.join(args.output_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels/val"), exist_ok=True)

def train_val_split(files, val_split):
    random.shuffle(files)
    split = int((1.0 - val_split) * len(files))
    train_files, val_files = files[:split], files[split:]
    tasks = []
    for idx, fname in enumerate(train_files, start=1):
        tasks.append((fname, idx, "train"))
    for idx, fname in enumerate(val_files, start=1):
        tasks.append((fname, idx, "val"))
    return tasks

def process_and_save(task):
    fname, idx, subset = task
    lr_dir = os.path.join(args.output_dir, "images/train") if subset == "train" else os.path.join(args.output_dir, "images/val")
    hr_dir = os.path.join(args.output_dir, "labels/train") if subset == "train" else os.path.join(args.output_dir, "labels/val")

    try:
        img = Image.open(os.path.join(args.input_dir, fname)).convert('RGB')
        if img.width < 1024 and img.height < 1024:
            hr = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
            w4 = (hr.width // 4) * 4
            h4 = (hr.height // 4) * 4
            if w4 < 4 or h4 < 4:
                return
            hr = hr.resize((w4, h4), Image.BICUBIC)
            lr = hr.resize((w4 // 4, h4 // 4), Image.BICUBIC)
            buf = io.BytesIO()
            lr.save(buf, format='JPEG', quality=50)
            buf.seek(0)
            lr = Image.open(buf)
            hr.save(os.path.join(hr_dir, f"{idx}.png"), icc_profile=None)
            lr.save(os.path.join(lr_dir, f"{idx}.png"), icc_profile=None)
    except Exception as e:
        pass

def main():
    create_dir()
    files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    tasks = train_val_split(files, args.val_split)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_and_save, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()