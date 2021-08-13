import enum
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from lxml import etree
from svg.path import parse_path
from tqdm import tqdm

def svg2masks(svg_file):
    svg_file = etree.parse(svg_file)
    root = svg_file.getroot()

    #Final all classes with name 'Wall External'
    wall = []
    for i in root.findall(".//{*}g[@class='Door Swing Beside']"):
        for j in i:
            if(j.get("points") != None):
                wall.append(j.get("points"))
    
    return wall

def pure_wall_image(root_dir):
    path, dir, files = next(os.walk(root_dir))
    os.chdir(root_dir + "wall_external_masked/")
    path,dir,files = next(os.walk(os.getcwd()))
    
    print("Initiating saving pure wall data\n")
    print("Saving...")
    for i in tqdm(enumerate(files)):
        external_wall_image = plt.imread(root_dir + "wall_external_masked/" + i[1])
        external_wall_image = cv2.cvtColor(external_wall_image,cv2.COLOR_BGR2GRAY)
        if(os.path.isfile(root_dir + "window_masked/" + i[1]) == True):
            window_image = plt.imread(root_dir + "window_masked/" + i[1])
            window_image = cv2.cvtColor(window_image,cv2.COLOR_BGR2GRAY)
        else:
            window_image = np.zeros([np.shape(external_wall_image)[0],np.shape(external_wall_image)[1]],dtype=np.float32)
        if(os.path.isfile(root_dir + "Wall_masked/" + i[1]) == True):
            internal_wall_image = plt.imread(root_dir + "Wall_masked/" + i[1])
            internal_wall_image = cv2.cvtColor(internal_wall_image,cv2.COLOR_BGR2GRAY)
        else:
            internal_wall_image =  np.zeros([np.shape(external_wall_image)[0],np.shape(external_wall_image)[1]],dtype=np.float32)
        if(os.path.isfile(root_dir + "doors/" + i[1]) == True):
            door_image = plt.imread(root_dir + "doors/" + i[1])
            door_image = cv2.cvtColor(door_image,cv2.COLOR_BGR2GRAY)
        else:
            door_image =  np.zeros([np.shape(external_wall_image)[0],np.shape(external_wall_image)[1]],dtype=np.float32)

        
        window_removed_image = cv2.subtract(external_wall_image, window_image)
        door_removed_image = cv2.subtract(window_removed_image,door_image)
        pure_wall = cv2.addWeighted(door_removed_image,1.0,internal_wall_image,1.0,0.0)
        pure_wall = pure_wall > 0

        plt.imsave("/home/aditya/Documents/HiWi/Work/cubicasa5k/Pure_wall/" + i[1],pure_wall)
    
    print("Saving pure wall data complete!")
    

"""
root_dir = "/home/aditya/Documents/HiWi/Work/cubicasa5k/colorful/"
path, dirs, files = next(os.walk(root_dir))
for k in tqdm(range(len(dirs))):
    os.chdir(root_dir + dirs[k])
    image_path = root_dir + dirs[k] + "/F1_scaled.png"
    image = plt.imread(image_path)
    image = np.zeros([np.shape(image)[0],np.shape(image)[1]])

    svg_file = root_dir + dirs[k] + "/model.svg"
    wall = svg2masks(svg_file)
    for i in range(len(wall)):
        wall_points = wall[i]
        wall_list = list(wall_points.split(" "))
        wall_list = wall_list[:-1]
    
        xval = []
        yval = []
        for j in range(len(wall_list)):
            a = wall_list[j]
            s = a.split(",")
            xval.append(int(float(s[0])))
            yval.append(int(float(s[1])))
    
        pts = np.array([[xval[0],yval[0]],[xval[1],yval[1]],[xval[2],yval[2]],[xval[3],yval[3]]],dtype=np.int32)
        image_masked = cv2.fillPoly(image,[pts],255)

        plt.imsave("/home/aditya/Documents/HiWi/Work/cubicasa5k/doors/" + dirs[k] + ".png",image_masked)
"""

root_dir = "/home/aditya/Documents/HiWi/Work/cubicasa5k/"
pure_wall_image(root_dir)
