import pandas as pd

files = {
    "test": "Final/data/paper_subset/test (1).csv",
    "train": "Final/data/paper_subset/train.csv",
    "validation": "Final/data/paper_subset/validation (2).csv",
}

dfs = {}
for name, path in files.items():
    df = pd.read_csv(path)
    dfs[name] = df
    print(f"\n=== {name} === ({len(df)} rows)")
    print(df.head(2))

# 1. Check duplicate IDs within each file
print("\n--- Duplicate IDs within each file ---")
for name, df in dfs.items():
    dupes = df[df.duplicated(subset=["id"], keep=False)]
    if dupes.empty:
        print(f"{name}: No duplicate IDs")
    else:
        print(f"{name}: {len(dupes)} rows with duplicate IDs:")
        print(dupes[["id", "text"]].to_string())

# 2. Check duplicate IDs across files
print("\n--- Duplicate IDs across files ---")
all_ids = []
for name, df in dfs.items():
    for id_val in df["id"]:
        all_ids.append((id_val, name))

from collections import defaultdict
id_map = defaultdict(list)
for id_val, name in all_ids:
    id_map[id_val].append(name)

cross_dupes = {k: v for k, v in id_map.items() if len(v) > 1}
if cross_dupes:
    print(f"Found {len(cross_dupes)} IDs appearing in multiple files:")
    for id_val, sources in cross_dupes.items():
        print(f"  ID {id_val}: {sources}")
else:
    print("No cross-file duplicate IDs found.")

# 3. Check if each sentence has only 1 aspect (single-aspect sentences)
print("\n--- Multi-aspect check (sentences with same id appearing more than once) ---")
for name, df in dfs.items():
    multi = df.groupby("id").filter(lambda x: len(x) > 1)
    if multi.empty:
        print(f"{name}: All sentences are single-aspect")
    else:
        print(f"{name}: {multi['id'].nunique()} sentences have multiple aspects:")
        print(multi.to_string())
