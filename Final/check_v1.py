import csv
from collections import Counter

file_path = 'data/v1.csv'
aspect_counts = Counter()
total_rows = 0

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        aspect = row['topic'].strip() if 'topic' in row else row.get('aspect', '').strip()
        sentiment = row['sentiment'].strip()
        aspect_counts[(aspect, sentiment)] += 1
        total_rows += 1

print(f"Total rows in v1.csv: {total_rows}")
for (aspect, sentiment), count in aspect_counts.most_common():
    print(f"  {aspect} ({sentiment}): {count}")

