import torch
import torchvision
from tqdm import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: str):
    # Put model in train mode
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        # Put data on target device
        x, y = x.to(device), y.to(device)

        # 1. Forward pass
        y_logits = model(x)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

        # 2. Calculate loss & accumulate
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward

        # 5. Optimizer step
        optimizer.step()

        # 6. Accumulate accuracy
        train_acc += (y_preds==y).sum().item()/len(y_preds)

    # Calculate loss & accuracy
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):
    # Put model in eval mode
    model.eval()

    test_loss, test_acc = 0, 0

    # No gradients
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            # Put data on target device
            x, y = x.to(device), y.to(device)

            # 1. Forward pass
            y_logits = model(x)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

            # 2. Calculate loss & accumulate
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()

            # 3. Accumulate accuracy
            test_acc += (y_preds==y).sum().item()/len(y_preds)
        
        # Calculate loss & accuracy
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)

        return test_loss, test_acc
    
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str,
          epochs: int):
    # Create results dictionary
    results = {"train_loss" : [],
               "train_acc" : [],
               "test_loss" : [],
               "test_acc" : []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results