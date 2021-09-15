import torch
import torch.nn as nn
def train_one_epoch(training_dataloader, criterion, model):
    running_loss = 0.0
    for i, data in enumerate(training_dataloader,0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
    return running_loss

def train(training_dataloader, criterion, model, epochs):
    for epoch in epochs:
        running_loss = train_one_epoch(training_dataloader, criterion, model)
        if epoch%10 = 9:
            print('[%d] loss: %.3f'%(epoch + 1, running_loss))
         
    return model
    
