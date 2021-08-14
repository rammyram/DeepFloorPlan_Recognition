import cv2
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def resize_images(source_dir1):
    path, dirs, files = next(os.walk(source_dir1))
    
    for i in range(len(files)):
        image = Image.open(source_dir1 + files[i]).convert("L")
        image = image.resize((600,600),Image.ANTIALIAS)
        image.save("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/windows/windows_val_new/" + files[i])

def change_mask_color(source_dir,destination_dir):
    path, dirs, files = next(os.walk(source_dir))
    path, dirs, files = next(os.walk(source_dir))
    file_1 = []
    for i in range(len(files)):
        files[i] = files[i].split("_")
        file_1.append(int(files[i][0]))
    file_1 = sorted(file_1)

    for i in range(len(file_1)):
        image = np.array(cv2.imread(source_dir + str(file_1[i]) + "_doors.png"))
        
        image[np.all(image == (255.0, 255.0,255.0),axis=-1)] = (0.0, 0.0, 255.0)
        cv2.imwrite(destination_dir + str(file_1[i]) + "_doors.png",image)


def merge_images(source_dir1, source_dir2):
    path, dirs, files = next(os.walk(source_dir1))
    file_1 = []
    for i in range(len(files)):
        files[i] = files[i].split("_")
        file_1.append(int(files[i][0]))
    file_1 = sorted(file_1)
    
    
    path, dirs, files = next(os.walk(source_dir2))
    file_2 = []
    for i in range(len(files)):
        files[i] = files[i].split("_")
        file_2.append(int(files[i][0]))
    file_2 = sorted(file_2)

    
    for i in tqdm(range(len(file_1))):
        src_img = plt.imread(source_dir1 + str(file_1[i]) + "_wall.png")
        src_img = cv2.resize(src_img,(600,600))
        src_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
        src_img[src_img == (1.0,1.0,1.0)] = 255
        src_img[:,:,1] = src_img[:,:,2] = 0
        
        dest_img = plt.imread(source_dir2 + str(file_2[i]) + "_doors.png")
        dest_img = cv2.resize(dest_img,(600,600))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB)
        dest_img[dest_img == 1.0] = 255
        dest_img[:,:,0] = dest_img[:,:,2] = 0

        
        new_img = cv2.addWeighted(src_img, 1.0, dest_img, 1.0, 0.0)
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/new_train/" + str(file_1[i]) + ".png",new_img)
  
def dimension_check(source_file):
    path, dirs, files = next(os.walk(source_file))
    
    for i in range(len(files)):
        image = Image.open(path + files[i]).convert("RGB")
        image = image.resize((600,600),Image.ANTIALIAS)
        print("Dimension of image: " + str(path) + str(files[i]) + " : " + str(np.shape(image)))
    
    print("Number of images checked: ",len(files))


def create_ground_truth_images(source_dir,destination_dir):
    path, dirs, files = next(os.walk(source_dir))

    for i in range(len(files)):
        if(files[i].endswith(".png")):
            image = Image.open(source_dir + str(files[i]))
            image = ImageOps.invert(image)
            image.save(destination_dir + str(files[i]))
    print("Completed creating initial version of ground truth images")


def seperate_wall_images(source_dir,destination_dir):
    path, dirs, files = next(os.walk(source_dir))

    for i in range(len(files)):
        if(files[i].endswith('_close_wall.png')):
            image = Image.open(source_dir + str(files[i]))
            image = image.resize((600,600),Image.ANTIALIAS)
            image.save(destination_dir + str(files[i]))
    
    print("Completed traferring wall images into a seperate folder")

def convert_images_to_rgb(source_dir):
    path, dirs, files = next(os.walk(source_dir))
    for i in tqdm(enumerate(files)):
        image = plt.imread(source_dir + i[1])
        image = cv2.resize(image, (256,256))
        if(np.shape(image) == (256,256,3)):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/Images/Images_val_new/" + i[1],image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/Images/Images_val_new/" + i[1],image)
    
    print("Successful")

#src_dir2 = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/doors/doors_train/"
src_dir1 = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/Images/Images_val/"
#destination_dir = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/Walls/Close_wall/"
#merge_images(src_dir1,src_dir2)
#dimension_check(src_dir1)
#resize_images(src_dir2)
#change_mask_color(src_dir1,destination_dir)
#seperate_wall_images(src_dir1,destination_dir)
convert_images_to_rgb(src_dir1)
