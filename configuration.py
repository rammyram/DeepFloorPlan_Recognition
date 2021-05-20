import torch

class training_config():
    batch_size = 2
    number_epochs = 200
    learning_rate = 0.0001
    number_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_frequency = 1

class train_data_config():
    training_set_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/Images/Image_train/"
    doors_training_ground_truth_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/doors/doors_train/"
    windows_training_ground_truth_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/windows/windows_train/"
    training_data_size = 146

class validation_data_config():
    validation_set_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/Images/Imgaes_val/"
    doors_validation_ground_truth_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/doors/doors_val/"
    windows_val_groud_truth_dir = "/home/aditya/Documents/HiWi/Work/Deep_Floor_Plan_Recognition/Code/DeepFloorPlan_Recognition/R3D_doors and windows/Train/windows/windows_val/"
    validation_data_size = 27

