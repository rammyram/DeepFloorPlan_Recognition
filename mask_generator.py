import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def xml_to_masks(xml_file):
    """
    The function takes in the xml file and returns the coordinates of the bounding box to be used to create masks
    """

    xml_file = ET.parse(xml_file)
    root = xml_file.getroot()

    points_min = []
    points_max = []
    r1 = root.findall('object')
    room_elements = []
    for i in r1:
        room_elements.append(ET.tostring(i)[17])

    for i in r1:
        if ET.tostring(i)[17] == 100: #100 - doors,119 - windows
            regions = i.findall('bndbox')
            
            for region in regions:
                for point in region.findall('xmin'):
                    if ET.tostring(point)[9] == 60:
                        xmin = int(ET.tostring(point)[6:9])
                    elif ET.tostring(point)[8] == 60:
                        xmin = int(ET.tostring(point)[6:8])
                    else:
                        xmin = int(ET.tostring(point)[6])

                    for point in region.findall('ymin'):
                        if ET.tostring(point)[9] == 60:
                            ymin = int(ET.tostring(point)[6:9])
                        elif ET.tostring(point)[8] == 60:
                            ymin = int(ET.tostring(point)[6:8])
                        else:
                            ymin = int(ET.tostring(point)[6])
                
                        points_min.append([xmin,ymin])
                    
            
                        for point in region.findall('xmax'):
                            if ET.tostring(point)[9] == 60:
                                xmax = int(ET.tostring(point)[6:9])
                            elif ET.tostring(point)[8] == 60:
                                xmax = int(ET.tostring(point)[6:8])
                            else:
                                xmax = int(ET.tostring(point)[6])

                            for point in region.findall('ymax'):
                                if ET.tostring(point)[9] == 60:
                                    ymax = int(ET.tostring(point)[6:9])
                                elif ET.tostring(point)[8] == 60:
                                    ymax = int(ET.tostring(point)[6:8])
                                else:
                                    ymax = int(ET.tostring(point)[6])
                                points_max.append([xmax,ymax])
    
    return(points_min,points_max)

root_dir = "/home/aditya/Documents/HiWi/Work/data/d1_sym/"
path, dors, files = next(os.walk(root_dir))
file_1 = []
for i in range(len(files)):
    if(files[i].endswith('.xml')):
        file_1.append(int(files[i][:-4]))
xml_files = sorted(file_1)

for i in tqdm(range(len(xml_files))):
    min_points, max_points = xml_to_masks(root_dir + str(xml_files[i]) + ".xml")
    image = plt.imread(root_dir + str(xml_files[i]) + ".gif")
    image = np.zeros([np.shape(image)[0],np.shape(image)[1]])
    for j in range(len(min_points)):
        image_1 = cv2.rectangle(image,tuple(min_points[j]),tuple(max_points[j]),(255),-1)
    plt.imsave("/home/aditya/Documents/HiWi/Work/data/d1_sym_masks_doors/" + str(xml_files[i]) + ".png",image_1)
