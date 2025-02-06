import os
import argparse

parser = argparse.ArgumentParser(description="Prepare and Train Script")
parser.add_argument("--data_process_folder", type=str, default="data_process", help="Path to the data processing folder")
parser.add_argument("--data_folder", type=str, default="data_process/dataset/output", help="Path to the dataset folder")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--weights_folder", type=str, default="models/weights", help="Folder to save model weights")
parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda:0"], help="Device to use for training")
args = parser.parse_args()

#preparation
ori_dir = os.getcwd()

os.chdir(args.data_process_folder)
os.system(f'python auto_process.py')

#train
os.chdir(ori_dir)

train_command = (
    f"python train_only.py "
    f"--data_folder {args.data_folder} "
    f"--batch_size {args.batch_size} "
    f"--learning_rate {args.learning_rate} "
    f"--num_epochs {args.num_epochs} "
    f"--weights_folder {args.weights_folder} "
    f"--device {args.device}"
)

os.system(train_command)