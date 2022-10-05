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

def linearize(data, type):
    ls = []

    if type == 0:
        for s in data:
            d = split_syllables(s)
            ls.append(d)
    else:
        for s in data:
            i = s[::-1]
            d = split_syllables(i)
            ls.append(d)
    
    return np.array(ls)

def process_splitted(first, second):
    first_np = linearize(first, 0)
    second_np = linearize(second, 1)

    print('linearized complete!')

    return first_np, second_np

def split_syll(sentence):
    splitted = split_syllables(sentence)
    return splitted

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
    ch2idx['<unk>'] = 1

    idx = 2
    idx, max_len, first_ls = tok(first, ch2idx, idx, max_len)
    idx, max_len, second_ls = tok(second, ch2idx, idx, max_len)
           
    print('done tokenizing both data!')

    record.recordInfo('ch2idx', ch2idx)

    return first_ls, second_ls, ch2idx, max_len

def enc(data, ch2idx, max_len, stat):
    ls = []

    for s in data:

        # if stat == 1:
        #     s = (['<pad>'] * (max_len - len(s))) + s
        # else:
        #     s += ['<pad>'] * (max_len - len(s))

        # doesn't matter for inverse
        s = (['<pad>'] * (max_len - len(s))) + s
            
        toked_id = [ch2idx.get(token, ch2idx['<unk>']) for token in s]

        ls.append(toked_id)
    
    return np.array(ls)

def encode(first, second, ch2idx, max_len):

    first2idx_np = enc(first, ch2idx, max_len, 1)
    second2idx_np = enc(second, ch2idx, max_len, 0)

    print('encoding comlete!')

    return first2idx_np, second2idx_np

def syll_enc(data, stat, max_len, ch2idx):

    # if stat == 1:
    #     data = (['<pad>'] * (max_len - len(data))) + data
    # else:
    #     data += ['<pad>'] * (max_len - len(data))

    # doesn't matter for inverse
    data = (['<pad>'] * (max_len - len(data))) + data
    
    toked_id = [ch2idx.get(token, ch2idx['<unk>']) for token in data]

    return np.array(toked_id)