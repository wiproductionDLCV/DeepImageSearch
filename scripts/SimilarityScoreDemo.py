
import os
import yaml
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path

# Add the current working directory (where your local DeepImageSearch is located)
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import DeepImageSearch
print(DeepImageSearch.__file__)
from DeepImageSearch import Load_Data,Search_Setup


# Load images from a folder
image_list = Load_Data().from_folder(['/home/wi/work/Dataset_animals/crops'])
unannotated_image_folder = '/home/wi/work/Dataset_animals/crop_nonannoted'
output_csv = '/home/wi/work/annotation_transfer_tool_gpu/result_sequential3.csv'
threshold = 0.2
# Set up the search engine
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True,use_gpu=True,use_batch_processing=True)

# Index the images
st.run_index()

## Extract feature from unanntated folder 
# st.annotate_folder(unannotated_image_folder, output_csv, threshold)
st.annotate_folder_sequential(unannotated_image_folder, output_csv, threshold)
