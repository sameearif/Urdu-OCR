import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--epochs", type=str, required=True)
parser.add_argument("--batch_size", type=str, required=True)
parser.add_argument("--imgsz", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--device", type=str, required=True)

args = parser.parse_args()

def main():
    model = YOLO(args.model)

    model.train(
        data=args.config,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch_size),
        name=args.output_file,
        device=args.device,
        save=True,
        save_period=-1,
        val=True,
        patience=50
    )

if __name__ == "__main__":
    main()