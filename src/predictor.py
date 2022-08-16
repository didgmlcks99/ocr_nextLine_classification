import torch
import torch.nn.functional as F
import utils
import data

def predict(first, second, model):

    first_syll = utils.split_syll(first)
    second_syll = utils.split_syll(second)

    first = utils.split(first_syll)
    second = utils.split(second_syll)

    ch2idx = data.getCh2idx()

    max_len = max(len(first), len(second))

    first = utils.syll_enc(first, 1, max_len, ch2idx)
    second = utils.syll_enc(second, 0, max_len, ch2idx)

    first = torch.tensor(first).unsqueeze(dim=0)
    second = torch.tensor(second).unsqueeze(dim=0)

    model.cpu()

    logits = model.forward(first, second)
    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    print(f"{probs[0]*100:.2f}% sure that the next line must be omitted.")