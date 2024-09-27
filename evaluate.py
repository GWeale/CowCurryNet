import torch
from model import get_model
from data_loader import get_data_loaders
from config import config
import os
import utils
import torch.nn as nn

def evaluate_model():
    _, val_loader = get_data_loaders()
    model = get_model(config.num_classes)
    model_path = os.path.join(config.model_save_path, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="evaluating"):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    total_loss = running_loss / len(val_loader.dataset)
    total_acc = running_corrects.double() / len(val_loader.dataset)
    utils.log(f"evaluation loss {total_loss:.4f} acc {total_acc:.4f}")
