# example run

# $env:DEEPINFRA_API_KEY=""
# $env:OPENAI_API_KEY=""
# $env:GOOGLE_API_KEY=""

## Winners
# python run_eval.py --provider openai --model_name gpt-4o
# python run_eval.py --provider openai --model_name gpt-4.1
# python run_eval.py --provider deepinfra --model_name meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
# python run_eval.py --provider deepinfra --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct
# python run_eval.py --provider gemini --model_name gemini-2.5-pro-preview-05-06

# Wildcards
# python run_eval.py --provider deepinfra --model_name Qwen/QVQ-72B-Preview
# python run_eval.py --provider deepinfra --model_name microsoft/Phi-4-multimodal-instruct
# python run_eval.py --provider deepinfra --model_name meta-llama/Llama-3.2-90B-Vision-Instruct

import os
import re
import json
import argparse
from models import TextRecognition
from jiwer import wer, cer

parser = argparse.ArgumentParser()
parser.add_argument("--provider", choices=["openai", "deepinfra", "gemini"], required=True)
parser.add_argument("--model_name", required=True)
args = parser.parse_args()

# setup
## provider-specific API URLs and keys
if args.provider == "openai":
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_URL = "https://api.openai.com/v1/chat/completions"
elif args.provider == "deepinfra":
    API_KEY = os.getenv("DEEPINFRA_API_KEY")
    API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
elif args.provider == "gemini":
    API_KEY = os.getenv("GOOGLE_API_KEY")
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{args.model_name}:generateContent"
else:
    raise ValueError("Unsupported provider")

if not API_KEY:
    raise ValueError(f"Missing API key for {args.provider}. Set it in your environment.")

## model class
MODEL_NAME = args.model_name
recognizer = TextRecognition(api_url=API_URL, api_key=API_KEY, model=MODEL_NAME, provider=args.provider)

## data paths
test_dir = "dataset/test"
gt_dir = "dataset/test-gt"
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

# evaluate
results = recognizer.evaluate(test_dir, gt_dir, file_list=image_files)

# save res
safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', MODEL_NAME)
output_path = f"ocr_results_{args.provider}_{safe_name}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"OCR results saved to {output_path}")
