import gensim
import numpy as np

def load_pretrained_model(ch2idx):
    word2vec = gensim.models.Word2Vec.load('../../../pretrained_model/kor/ko.bin')

    embeddings = np.random.uniform(-0.25, 0.25, (len(ch2idx), 200))
    embeddings[ch2idx['<pad>']] = np.zeros((200,))
    
    count = 0
    for ch in ch2idx:
        if ch in word2vec:
            count += 1
            embeddings[ch2idx[ch]] = word2vec[ch]
    
    print(f"There are {count} / {len(ch2idx)} pretrained vectors found.")

    return embeddings