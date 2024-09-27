import argparse
from train import train_model
from evaluate import evaluate_model
from utils import setup_logging, log

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.train:
        log("training initiated")
        train_model()
    if args.evaluate:
        log("evaluation initiated")
        evaluate_model()
    if not (args.train or args.evaluate):
        print("specify --train or --evaluate")

if __name__ == '__main__':
    main()
