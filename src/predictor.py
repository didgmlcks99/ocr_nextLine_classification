import torch
import torch.nn.functional as F
import utils
import data
import data_related

def predict(first, second, model, sORr):

    first_syll = utils.split_syll(first)
    
    second_inverted = second[::-1]
    second_syll = utils.split_syll(second_inverted)

    first = utils.split(first_syll)
    second = utils.split(second_syll)

    if sORr:
        ch2idx = data.getCh2idx()
    else:
        ch2idx = data_related.getCh2idx()

    max_len = max(len(first), len(second))

    first = utils.syll_enc(first, 1, max_len, ch2idx)
    second = utils.syll_enc(second, 0, max_len, ch2idx)

    first = torch.tensor(first).unsqueeze(dim=0)
    second = torch.tensor(second).unsqueeze(dim=0)

    # print(first)
    # print(second)

    model.cpu()

    logits = model.forward(first, second)
    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    print(f"{probs[0]*100:.2f}% that next line must be omitted.")

    result = logits.argmax(1).unsqueeze(1)

    pritn('result: ' + str(result))