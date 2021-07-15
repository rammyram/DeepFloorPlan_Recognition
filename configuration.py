import torch

class training_config():
    batch_size = 2
    number_epochs = 25
    learning_rate = 0.000001
    number_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_frequency = 5

class train_data_config():
    training_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_train/"
    train_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/train_no_seperation/"
    training_data_size = 40

class validation_data_config():
    validation_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_val/"
    validation_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/val_no_seperation/"
    validation_data_size = 10

