import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

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
        
        image[np.all(image == (255.0, 255.0,255.0),axis=-1)] = (128.0, 0.0, 128.0)
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

    
    for i in range(len(file_1)):
        src_img = cv2.imread(source_dir1 + str(file_1[i]) + "_doors.png")
        dest_img = cv2.imread(source_dir2 + str(file_2[i]) + "_windows.png")

        src_img = Image.fromarray(src_img)
        dest_img = Image.fromarray(dest_img)

        new_img = Image.blend(src_img,dest_img,0.5)
        new_img.save("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/new_val/" + str(file_1[i]) + ".png")

"""
src_image = cv2.imread("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/Images/Images_train/2.jpg")
src_image = Image.fromarray(src_image)
destination_image = cv2.imread("/home/aditya/Documents/HiWi/Work/R3D_doors and windows/new_train/2.png")
destination_image = Image.fromarray(destination_image)
new_image = Image.blend(src_image,destination_image,0.5)
new_image.show()
"""     

src_dir1 = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/doors/doors_val_color/"
src_dir2 = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/windows/windows_val_new/"
#destination_dir = "/home/aditya/Documents/HiWi/Work/R3D_doors and windows/Train/doors/doors_val_color/"
merge_images(src_dir1,src_dir2)
#resize_images(src_dir2)
#change_mask_color(src_dir1,destination_dir)

