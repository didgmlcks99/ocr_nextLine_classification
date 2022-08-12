import torch
import torch.nn as nn
import numpy as np

def test(
    test_dataloader=None,
    device=None, 
    model=None):
    
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    tot_loss = []
    tot_pred = torch.empty(0)
    tot_label = torch.empty(0)

    tot_pred = tot_pred.to(device)
    tot_label = tot_label.to(device)

    for batch in test_dataloader:
        firsts, seconds, labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(firsts, seconds)
        
        loss = loss_fn(logits, labels)
        tot_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        tot_pred = torch.cat((tot_pred, preds))
        tot_label = torch.cat((tot_label, labels))
    
    fin_loss = np.mean(tot_loss)
    fin_acc = (tot_pred == tot_label).cpu().numpy().mean() * 100
    
    print('test loss: ', fin_loss)
    print('test acc: ', fin_acc)

    return tot_pred, 