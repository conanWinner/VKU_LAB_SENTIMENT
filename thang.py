import csv
import ast
import os

input_path = '/home/conanwinner/Desktop/_CODE/VKU_Lab_Sentiment/data_1 - neutral.csv'
output_path = '/home/conanwinner/Desktop/_CODE/VKU_Lab_Sentiment/data_1 - neutral_expanded.csv'

rows = []
with open(input_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            sentences = ast.literal_eval(row['augmented_sentences'])
        except Exception:
            sentences = [row['augmented_sentences']]
        for sent in sentences:
            rows.append({
                'Sentence': row['Sentence'],
                'Aspect Term': row['Aspect Term'],
                'polarity': row['polarity'],
                'augmented_sentences': sent
            })

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Sentence', 'Aspect Term', 'polarity', 'augmented_sentences'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done: {len(rows)} rows written to {output_path}")