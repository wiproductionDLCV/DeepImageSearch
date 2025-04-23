import os
import shutil
import sys
sys.path.append('/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch')

from DeepImageSearch.DeepImageSearch import Load_Data, Search_Setup

# class CuratorImageClassifier:
#     def __init__(self, model_name='vgg19', threshold=0.7):
#         self.threshold = threshold
#         self.model_name = model_name
#         self.category_base_path = '/media/shubham/ssd_hub1/neha_work/crop_grounded_truth/General_Object_Category'
#         self.temp_folder = '/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch/temp'

#         # Load the base category images
#         image_list = Load_Data().from_folder([self.category_base_path])
#         self.st = Search_Setup(image_list=image_list, model_name=self.model_name, pretrained=True)
#         self.st.run_index()

#     def classify_bulk_images(self, bulk_folder_path, selected_category):
#         # Validate selected category
#         category_folder = os.path.join(self.category_base_path, selected_category)
#         print(f"\nAvailable categories: {os.listdir(self.category_base_path)}")
#         print(f"Selected category: {selected_category}")

#         if not os.path.isdir(category_folder):
#             raise ValueError(f"Invalid category selected: {selected_category}")

#         # Get all image paths from the bulk folder
#         all_images = [
#             os.path.join(bulk_folder_path, fname)
#             for fname in os.listdir(bulk_folder_path)
#             if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ]

#         print(f"Scanned {len(all_images)} image(s) in {bulk_folder_path}")
#         print("Image files found:", all_images)

#         print(f"\nTotal images to classify: {len(all_images)}")

#         for idx, img_path in enumerate(all_images, start=1):
#             image_name = os.path.basename(img_path)
#             print(f"\n[{idx}/{len(all_images)}] Processing image: {image_name}")

#             is_similar = self.st.get_similar_images_curator(image_path=img_path, threshold=self.threshold)

#             if is_similar:
#                 os.makedirs(category_folder, exist_ok=True)
#                 target_path = os.path.join(category_folder, image_name)
#                 shutil.move(img_path, target_path)
#                 print(f"Image '{image_name}' moved to category folder: {selected_category}")
#             else:
#                 os.makedirs(self.temp_folder, exist_ok=True)
#                 target_path = os.path.join(self.temp_folder, image_name)
#                 shutil.move(img_path, target_path)
#                 print(f"Image '{image_name}' did not meet similarity threshold. Moved to TEMP folder.")

# if __name__ == "__main__":
#     curator = CuratorImageClassifier(threshold=0.7)
#     # Selected from GUI dropdown
#     selected_category = "Pot" 
#     bulk_folder = "/media/shubham/ssd_hub1/neha_work/crop_synthetic/General_Object_Category/Pot"
#     curator.classify_bulk_images(bulk_folder, selected_category)


# class CuratorImageClassifier:
#     def __init__(self, model_name='vgg19', threshold=0.7):
#         self.threshold = threshold
#         self.model_name = model_name
#         self.category_base_path = '/media/shubham/ssd_hub1/neha_work/crop_grounded_truth/General_Object_Category'
#         self.temp_folder = '/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch/temp'

#         # Load the base category images
#         image_list = Load_Data().from_folder([self.category_base_path])
#         self.st = Search_Setup(image_list=image_list, model_name=self.model_name, pretrained=True)
#         self.st.run_index()

#     def classify_bulk_images(self, bulk_folder_path, selected_category):
#         # Validate selected category
#         category_folder = os.path.join(self.category_base_path, selected_category)
#         print(f"\nAvailable categories: {os.listdir(self.category_base_path)}")
#         print(f"Selected category: {selected_category}")

#         if not os.path.isdir(category_folder):
#             raise ValueError(f"Invalid category selected: {selected_category}")

#         # Get all image paths from the bulk folder
#         all_images = [
#             os.path.join(bulk_folder_path, fname)
#             for fname in os.listdir(bulk_folder_path)
#             if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ]

#         print(f"Scanned {len(all_images)} image(s) in {bulk_folder_path}")
#         print("Image files found:", all_images)

#         print(f"\nTotal images to classify: {len(all_images)}")

#         for idx, img_path in enumerate(all_images, start=1):
#             image_name = os.path.basename(img_path)
#             print(f"\n[{idx}/{len(all_images)}] Processing image: {image_name}")

#             is_similar = self.st.get_similar_images_curator(image_path=img_path, threshold=self.threshold)

#             if is_similar:
#                 os.makedirs(category_folder, exist_ok=True)
#                 target_path = os.path.join(category_folder, image_name)
#                 shutil.move(img_path, target_path)
#                 print(f"Image '{image_name}' moved to category folder: {selected_category}")
#             else:
#                 os.makedirs(self.temp_folder, exist_ok=True)
#                 target_path = os.path.join(self.temp_folder, image_name)
#                 shutil.move(img_path, target_path)
#                 print(f"Image '{image_name}' did not meet similarity threshold. Moved to TEMP folder.")


# if __name__ == "__main__":
#     curator = CuratorImageClassifier(threshold=0.7)
#     selected_category = "foil"  # Selected from UI or dropdown
#     bulk_folder = "/media/shubham/ssd_hub1/neha_work/crop_synthetic/General_Object_Category/Pot"
#     curator.classify_bulk_images(bulk_folder, selected_category)

import os
import shutil
import sys
from PIL import Image  # For optional display of images

sys.path.append('/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch')
from DeepImageSearch.DeepImageSearch import Load_Data, Search_Setup

class CuratorImageClassifier:
    def __init__(self, model_name='vgg19', threshold=0.7, selected_category='Pot'):
        self.threshold = threshold
        self.model_name = model_name
        self.selected_category = selected_category

        self.category_base_path = '/media/shubham/ssd_hub1/neha_work/crop_grounded_truth/General_Object_Category'
        self.temp_folder = '/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch/temp'

        # Load only the selected category for indexing (e.g., "Pot")
        category_path = os.path.join(self.category_base_path, self.selected_category)
        if not os.path.exists(category_path):
            raise ValueError(f"Category path not found: {category_path}")
        
        image_list = Load_Data().from_folder([category_path])
        self.st = Search_Setup(image_list=image_list, model_name=self.model_name, pretrained=True)
        self.st.run_index()

    # def classify_bulk_images(self, bulk_folder_path):
    #     category_folder = os.path.join(self.category_base_path, self.selected_category)

    #     if not os.path.isdir(category_folder):
    #         raise ValueError(f"Invalid category selected: {self.selected_category}")

    #     all_images = [
    #         os.path.join(bulk_folder_path, fname)
    #         for fname in os.listdir(bulk_folder_path)
    #         if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    #     ]

    #     print(f"Scanned {len(all_images)} image(s) in {bulk_folder_path}")
    #     print("Image files found:", all_images)

    #     for idx, img_path in enumerate(all_images, start=1):
    #         image_name = os.path.basename(img_path)
    #         print(f"\n[{idx}/{len(all_images)}] Processing image: {image_name}")

    #         is_similar = self.st.get_similar_images_curator(image_path=img_path, threshold=self.threshold)

    #         if is_similar:
    #             os.makedirs(category_folder, exist_ok=True)
    #             target_path = os.path.join(category_folder, image_name)
    #             shutil.move(img_path, target_path)
    #             print(f"Image '{image_name}' moved to category folder: {self.selected_category}")
    #         else:
    #             os.makedirs(self.temp_folder, exist_ok=True)
    #             target_path = os.path.join(self.temp_folder, image_name)
    #             shutil.move(img_path, target_path)
    #             print(f"Image '{image_name}' did not meet similarity threshold. Moved to TEMP folder.")

    def classify_bulk_images(self, bulk_folder_path):
        category_folder = os.path.join(self.category_base_path, self.selected_category)
        #temp_folder = "/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch/temp"

        # Ensure the category folder exists
        if not os.path.isdir(category_folder):
            raise ValueError(f"Invalid category selected: {self.selected_category}")

        all_images = [
            os.path.join(bulk_folder_path, fname)
            for fname in os.listdir(bulk_folder_path)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"Scanned {len(all_images)} image(s) in {bulk_folder_path}")
        print("Image files found:", all_images)

        for idx, img_path in enumerate(all_images, start=1):
            image_name = os.path.basename(img_path)
            print(f"\n[{idx}/{len(all_images)}] Processing image: {image_name}")

            # Check if image already exists in the category folder
            category_image_path = os.path.join(category_folder, image_name)
            if os.path.exists(category_image_path):
                print(f"Image '{image_name}' already in category folder. Skipping.")
                continue

            # Get similarity result
            is_similar = self.st.get_similar_images_curator(image_path=img_path, threshold=self.threshold)

            if is_similar:
                # Move to category folder
                os.makedirs(category_folder, exist_ok=True)
                target_path = os.path.join(category_folder, image_name)
                shutil.move(img_path, target_path)
                print(f"Image '{image_name}' moved to category folder: {self.selected_category}")
            else:
                # Move to temporary folder
                os.makedirs(self.temp_folder, exist_ok=True)
                target_path = os.path.join(self.temp_folder, image_name)
                shutil.move(img_path, target_path)
                print(f"Image '{image_name}' did not meet similarity threshold. Moved to TEMP folder.")

if __name__ == "__main__":
    selected_category = "foil"
    bulk_folder = "/media/shubham/ssd_hub1/neha_work/crop_synthetic/General_Object_Category/foil"

    curator = CuratorImageClassifier(threshold=0.7, selected_category=selected_category)
    curator.classify_bulk_images(bulk_folder)
