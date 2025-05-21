### Sample Dataset Link:
### https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

import sys
import os

# Add the current working directory (where your local DeepImageSearch is located)
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import DeepImageSearch
print(DeepImageSearch.__file__)
from DeepImageSearch import Load_Data,Search_Setup


# Load images from a folder
image_list = Load_Data().from_folder(['/home/wi/work/Dataset_animals/crops'])

# Set up the search engine
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=5000,use_gpu=True,use_batch_processing=False)

# Index the images
st.run_index()

# # Get metadata
# metadata = st.get_image_metadata_file()

# # Add New images to the index
# st.add_images_to_index(['image_path_1', 'image_path_2'])

# # Get similar images
# st.get_similar_images(image_path='image_path', number_of_images=10)

# # Plot similar images
# st.plot_similar_images(image_path='image_path', number_of_images=9)

# # Update metadata
# metadata = st.get_image_metadata_file()