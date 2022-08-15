import utils
import data

def predict(first, second):

    first_syll = utils.split_syll(first)
    second_syll = utils.split_syll(second)

    ch2idx = data.getCh2idx()

    max_len = max(len(first), len(second))

    first = utils.split_enc(first, 1, max_len, ch2idx)
    second = utils.split_enc(first, 1, max_len, ch2idx)