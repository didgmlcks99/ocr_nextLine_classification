import ocr_model as om
import torch.optim as optim
import torch.nn as nn

def initialize_model(
    vocab_size=None,
    embed_dim=200,
    hidden_size=100,
    num_classes=2,
    rnn_layers=3,
    dropout=0.2,
    learning_rate=0.001,
    model_name="RNN",
    optim_name="Adam",
    loss_fn_name="CEL",
    pretrained_model=None,
    freeze_embedding=False,
    device=None,
    ):

    if model_name=="RNN":

        model = om.OCR(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            rnn_layers=rnn_layers,
            dropout=dropout,
            pretrained_model=pretrained_model,
            freeze_embedding=freeze_embedding
        )
    
    model.to(device)

    if optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    
    if loss_fn_name == "CEL":
        loss_fn = nn.CrossEntropyLoss()
    

    return model, optimizer, loss_fn