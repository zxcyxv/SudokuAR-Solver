from huggingface_hub import hf_hub_download
import csv

def check_size():
    repo_id = "sapientinc/sudoku-extreme"
    total = 0
    for split in ["train", "test"]:
        print(f"Checking {split}...")
        csv_path = hf_hub_download(repo_id, f"{split}.csv", repo_type="dataset")
        with open(csv_path, newline="") as f:
            # -1 for header
            count = sum(1 for _ in f) - 1
            print(f"{split}: {count:,}")
            total += count
    print(f"Total max samples: {total:,}")

if __name__ == "__main__":
    check_size()
