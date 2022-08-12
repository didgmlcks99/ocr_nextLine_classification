import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

def data_loader(train_first,
                train_second,
                val_first,
                val_second,
                test_first,
                test_second,
                train_labels,
                val_labels,
                test_labels,
                batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """

    # Convert data type to torch.Tensor
    train_first, train_second, val_first, val_second, test_first, test_second, train_labels, val_labels, test_labels =\
    tuple(torch.tensor(data) for data in
          [train_first,
           train_second,
           val_first,
           val_second,
           test_first,
           test_second,
           train_labels,
           val_labels,
           test_labels])

    # Specify batch_size
    batch_size = 50

    # Create DataLoader for training data
    train_data = TensorDataset(train_first, train_second, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_first, val_second, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=True)

    # Create DataLoader for training data
    test_data = TensorDataset(test_first, test_second, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader 