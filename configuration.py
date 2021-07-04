import torch

class training_config():
    batch_size = 2
    number_epochs = 1
    learning_rate = 0.00001
    number_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_frequency = 5

class train_data_config():
    training_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_train/"
    train_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/Walls_train/"
    training_data_size = 40

class validation_data_config():
    validation_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_val/"
    validation_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/Walls_val/"
    validation_data_size = 10

