import csv

def count_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            sentences = set()
            pairs = 0
            for row in reader:
                if len(row) >= 3:
                    sentences.add(row[0])
                    pairs += 1
            return len(sentences), pairs
    except Exception as e:
        return 0, 0

print("Train_1:", count_file('data/split/train_1.csv'))
print("Validation:", count_file('data/validation.csv'))
print("Test:", count_file('data/test.csv'))
