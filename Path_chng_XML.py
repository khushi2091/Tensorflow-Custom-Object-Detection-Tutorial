###### Please run the code from the directory wherever you would like to change the path of xml files ##########

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

path = os.getcwd()
path_new = r'C:\TF_object_detection\models\research\object_detection\hand_detector\images\train'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        #print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            xml_path = root.find('path').text
            temp = os.path.split(xml_path)[0]
            print(temp)
            temp = path_new
            root.find('path').text = os.path.join(temp, os.path.split(xml_path)[1])
            value = (root.find('folder').text, root.find('filename').text, 
                      root.find('path').text, member[0].text)
        tree.write(xml_file)

os.chdir(r'C:\TF_object_detection\models\research\object_detection\hand_detector\images\train')
xml_to_csv(path)