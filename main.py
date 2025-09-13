import os
import subprocess

MODEL_PATH = "models/pneumonia_model.h5"

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        subprocess.run(["python", "train_model.py"], check=True)
    else:
        print("Model already exists, skipping training.")

    print("Evaluating model...")
    subprocess.run(["python", "evaluate_model.py"], check=True)

