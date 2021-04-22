import torch

class training_config():
    training_batch_size = 2
    train_number_epochs = 
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_frequency = 1

class train_data_config():
    training_set_dir = ""
    training_ground_truth_dir = ""
    training_data_size = 

class validation_data_config():
    validation_set_dir = ""
    validation_ground_truth_dir = ""
    validation_data_size = 

