import json
import sys

def process():
    file_path = 'billstm_cnn.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                # source is a list of strings
                for i, line in enumerate(source):
                    if 'train_s =' in line and 'df_train1' in line:
                        # Rebuild this section
                        pass # actually, usually kaggle notebook API or newer Jupyter formats source as a single string or list of strings.
            elif isinstance(source, str):
                if 'train_s =' in source and 'df_train1' in source:
                    # Replace the entire assignment
                    old_code = "train_s = (load_bilstm_data(df_train1) + load_bilstm_data(df_train2) +\n           load_bilstm_data(df_train3) + load_bilstm_data(df_train4) +\n           load_bilstm_data(df_train5) + load_bilstm_data(df_train6))"
                    new_code = "train_s = load_bilstm_data(df_train1)"
                    if old_code in source:
                        cell['source'] = source.replace(old_code, new_code)
                        print("Replaced successfully.")
                    else:
                        print("Could not find the exact string to replace.")
                        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    process()
