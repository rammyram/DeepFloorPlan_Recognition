import torch

class training_config():
    batch_size = 2
    number_epochs = 200
    learning_rate = 0.0001
    number_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_frequency = 1

class train_data_config():
    training_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_train/"
    doors_training_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/doors_train/"
    windows_training_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/windows_train/"
    training_data_size = 146

class validation_data_config():
    validation_set_dir = "/content/DeepFloorPlan_Recognition/Data/Images_val/"
    doors_validation_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/doors_val/"
    windows_validation_ground_truth_dir = "/content/DeepFloorPlan_Recognition/Data/windows_val/"
    validation_data_size = 27

