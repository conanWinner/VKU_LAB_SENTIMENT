from datasets import load_dataset
import os
import csv

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    ds = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
    print(ds)

    for split in ["train", "validation", "test"]:
        if split in ds:
            output_json = os.path.join(SAVE_DIR, f"{split}.json")
            output_csv = os.path.join(SAVE_DIR, f"{split}.csv")

            ds[split].to_json(output_json, force_ascii=False)
            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(ds[split].column_names)
                for row in ds[split]:
                    writer.writerow([row[column] for column in ds[split].column_names])

            print(f"Saved {split} split to {output_json} and {output_csv} ({len(ds[split])} rows)")
        else:
            print(f"Split '{split}' not found in dataset")
