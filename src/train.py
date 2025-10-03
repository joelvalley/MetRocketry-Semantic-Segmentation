import torch
import torchvision

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