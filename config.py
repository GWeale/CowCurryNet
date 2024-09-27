import os
import torch

class config:
    data_dir = os.path.join(os.getcwd(), 'data')
    batch_size = 64
    num_workers = 8
    num_classes = 1000
    learning_rate = 0.0005
    momentum = 0.95
    weight_decay = 5e-4
    num_epochs = 100
    model_save_path = os.path.join(os.getcwd(), 'models')
    log_dir = os.path.join(os.getcwd(), 'logs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_step = 10
    scheduler_gamma = 0.1
    seed = 42
    image_size = 224
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
