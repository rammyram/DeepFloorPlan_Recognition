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
    for i in root.findall(".//{*}g[@class='Stairs']"):
        for j in i:
            if(j.get("points") != None):
                wall.append(j.get("points"))
    
    return wall

root_dir = "/home/aditya/Documents/HiWi/Work/cubicasa5k/high_quality/"
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

        plt.imsave("/home/aditya/Documents/HiWi/Work/cubicasa5k/stairs_masked/" + dirs[k] + ".png",image_masked)
