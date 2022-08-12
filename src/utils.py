from hangul_utils import split_syllables, join_jamos
import numpy as np
import record

def process(df):
    first_list = []
    second_list = []

    for i, row in df.iterrows():
        first = row['first']
        second = row['second']

        first = split_syllables(first)
        second = split_syllables(second)

        first_list.append(first)
        second_list.append(second)


    first_np = np.array(first_list)
    second_np = np.array(second_list)
    label_np = df['label'].to_numpy()

    return first_np, second_np, label_np

def linearize(data):
    ls = []

    for s in data:
        d = split_syllables(s)
        ls.append(d)
    
    return np.array(ls)

def process_splitted(first, second):
    first_np = linearize(first)
    second_np = linearize(second)

    return first_np, second_np

def split(str):
    return [char for char in str]

def tok(data, dict, idx, maxim):
    ls = []

    for s in data:
        toked = split(s)
        ls.append(toked)

        for key in toked:
            if key not in dict:
                dict[key] = idx
                idx += 1
        
        maxim = max(maxim, len(toked))
    
    return idx, maxim, ls

def tokenize(first, second):
    
    max_len = -1
    ch2idx = {}

    ch2idx['<pad>'] = 0

    idx = 1
    idx, max_len, first_ls = tok(first, ch2idx, idx, max_len)
    idx, max_len, second_ls = tok(second, ch2idx, idx, max_len)
           
    record.recordInfo('ch2idx', ch2idx)

    return first_ls, second_ls, ch2idx, max_len

def enc(data, ch2idx, max_len):
    ls = []

    for s in data:
        s += ['<pad>'] * (max_len - len(s))

        toked_id = [ch2idx.get(token) for token in s]

        ls.append(toked_id)
    
    return np.array(ls)

def encode(first, second, ch2idx, max_len):

    first2idx_np = enc(first, ch2idx, max_len)
    second2idx_np = enc(second, ch2idx, max_len)

    return first2idx_np, second2idx_np
