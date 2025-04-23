### Sample Dataset Link:
### https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

from DeepImageSearch import Load_Data, Search_Setup

# Load images from a folder
image_list = Load_Data().from_folder(['/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/crop/General_Object_Category'])

# Set up the search engine
st = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=100)

# Index the images
st.run_index()

# Get metadata
metadata = st.get_image_metadata_file()

# Add New images to the index
st.add_images_to_index(['/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/move_articles/crop_dataset/crop_20250108T163816_76_0_f20f72b8-e282-4659-81f2-71738acc466f.png'])

# Get similar images
# st.get_similar_images(image_path='/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/move_articles/crop_dataset/Screenshot from 2025-04-04 13-11-43.png', number_of_images=10)

# st.get_similar_images(image_path="/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/move_articles/crop_dataset/Screenshot from 2025-04-04 13-11-43.png", number_of_images=10)
st.get_similar_images_curator(image_path="/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/move_articles/crop_dataset/crop_20250108T163816_76_0_f20f72b8-e282-4659-81f2-71738acc466f.png", threshold=0.4)

# Plot similar images
#st.plot_similar_images(image_path='/media/shubham/ssd_hub1/hrithik_work/triton_ws/env/move_articles/crop_dataset/Screenshot from 2025-04-04 13-13-21.png', number_of_images=9)

# Update metadata
metadata = st.get_image_metadata_file()