from model import train_model_full
import argparse


def main(**kwargs):

    print("Executing full train")
    parser = argparse.ArgumentParser()
    parser.add_argument('--projectId', type=int, help='Label studio project ID')
    parser.add_argument('--epochs', type=int, help='Epochs to run training for', default=10)
    args = parser.parse_args()

    print(args.epochs)
    print(args.projectId)

    train_model_full(args.projectId, args.epochs)


if __name__ == "__main__":
    main()
