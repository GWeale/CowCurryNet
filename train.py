import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from model import get_model
from data_loader import get_data_loaders
from config import config
import os
import utils
import torch.nn as nn

def train_model():
    train_loader, val_loader = get_data_loaders()
    model = get_model(config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    best_acc = 0.0
    utils.log("starting training")
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"epoch {epoch+1} training"):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        utils.log(f"epoch {epoch+1} training loss {epoch_loss:.4f} acc {epoch_acc:.4f}")
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"epoch {epoch+1} validation"):
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        utils.log(f"epoch {epoch+1} validation loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(config.model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.model_save_path, 'best_model.pth'))
    utils.log(f"best validation accuracy {best_acc:.4f}")
