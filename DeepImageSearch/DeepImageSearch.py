import shutil
import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math
import faiss
import gc
import time
import csv
import matplotlib.pyplot as plt

class Load_Data:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass
    
    def from_folder(self, folder_list: list):
        """
        Adds images from the specified folders to the image_list.

        Parameters:
        -----------
        folder_list : list
            A list of paths to the folders containing images to be added to the image_list.
        """
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_path.append(os.path.join(root, file))
        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str):
        """
        Adds images from the specified column of a CSV file to the image_list.

        Parameters:
        -----------
        csv_file_path : str
            The path to the CSV file.
        images_column_name : str
            The name of the column containing the paths to the images to be added to the image_list.
        """
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        return pd.read_csv(self.csv_file_path)[self.images_column_name].to_list()

class Search_Setup:
    """ A class for setting up and running image similarity search."""
    def __init__(self, image_list: list, model_name='vgg19', pretrained=True, image_count: int = None,use_gpu=False,use_batch_processing=False):
        """
        Parameters:
        -----------
        image_list : list
        A list of images to be indexed and searched.
        model_name : str, optional (default='vgg19')
        The name of the pre-trained model to use for feature extraction.
        pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.
        image_count : int, optional (default=None)
        The number of images to be indexed and searched. If None, all images in the image_list will be used.
        """

        # Set device: GPU if available, else CPU
        self.use_gpu = use_gpu
        self.image_batch_size = 1000
        self.batch_size = 32

        if self.use_gpu : 
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            faiss_num_gpu = faiss.get_num_gpus() 
            print(f"\033[92m Number of GPUs available for FAISS: {faiss_num_gpu}")

        else: 
            self.device = 'cpu'
        print(f"\033[92m Using device: {self.device}")


        self.model_name = model_name
        self.pretrained = pretrained
        self.image_data = pd.DataFrame()
        self.d = None
        if image_count==None:
            self.image_list = image_list
        else:
            self.image_list = image_list[:image_count]

        self.use_batch_processing = use_batch_processing

        if self.use_batch_processing:
            print(f"\033[92m Using Batch processing to extract features from images")

        if f'metadata-files/{self.model_name}' not in os.listdir():
            try:
                os.makedirs(f'metadata-files/{self.model_name}')
            except Exception as e:
                # Handle the exception
                print(f'\033[91m An error occurred while creating the directory: metadata-files/{self.model_name}')
                print(f'\033[91m  Error Details: {e}')
        # Load the pre-trained model and remove the last layer
        print("\033[91m Please Wait Model Is Loading or Downloading From Server!")
        base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)  # Move model to GPU if available
        print(f"\033[92m Model Loaded Successfully: {model_name}")

    def _clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("\033[94m Cleared GPU memory cache")
            
    def _extract(self, img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(self.device) # [1, 3, 224, 224]

        # Extract features
        with torch.no_grad():
            feature = self.model(x)  # [1, F]

        feature = feature.detach().cpu().numpy().flatten()     # [F]  #  Feature matrix shape: (1000, 4096)
        
        return feature / np.linalg.norm(feature)
    
    

    def _extract_batch(self, img_list, batch_size=32):
        """
        Extracts normalized features from a list of PIL images in batches using GPU.
        Returns a list of 1D NumPy arrays (shape: [4096,]), like _extract.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        all_features = []
        all_tensors = [preprocess(img.convert("RGB")) for img in img_list]
        dataset = torch.stack(all_tensors)  # (N, 3, 224, 224)

        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size].to(self.device)
                outputs = self.model(batch)  # Shape: (B, F) or (B, F, 1, 1)

                # Flatten per-feature if needed
                if len(outputs.shape) > 2:
                    outputs = outputs.view(outputs.size(0), -1)  # (B, F)

                outputs = outputs.detach().cpu().numpy()

                # Normalize each vector and flatten to 1D
                normed_features = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
                flattened = [feat.flatten() for feat in normed_features]

                all_features.extend(flattened)

        return all_features  # List of arrays, each of shape (4096,)


    def _get_feature(self, image_data: list):
        self.image_data = image_data
        features = []
        for img_path in tqdm(self.image_data):  # Iterate through images
            # Extract features from the image
            try:
                feature = self._extract(img=Image.open(img_path))
                features.append(feature)
            except:
               # If there is an error, append None to the feature list
               print("\033[91m Failed to extract Feature from image")
               features.append(None)
               continue        

        return features
    
    def _get_feature_batch(self, image_data: list, image_batch_size=1000, feature_extraction_batch_size=32):
        """
        Efficient and scalable feature extraction:
        - Loads images in `image_batch_size` to avoid memory/file limits
        - Uses `feature_extraction_batch_size` for GPU batch inference
        """
        self.image_data = image_data
        total_images = len(image_data)
        full_features = []

        for i in tqdm(range(0, total_images, image_batch_size), desc="Extracting features for image batch "):
            batch_paths = image_data[i:i + image_batch_size]

            images = []
            valid_paths = []
            failed_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path)
                    img.load()
                    images.append(img.copy())
                    img.close()
                    valid_paths.append(path)
                except Exception as e:
                    print(f"\033[91m Failed to open image: {path} | Error: {e}")
                    failed_paths.append(path)

            # Extract features for the current image batch
            batch_features = self._extract_batch(images, batch_size=feature_extraction_batch_size)

            # Reconstruct full feature list
            valid_idx = 0
            for path in batch_paths:
                if path in valid_paths:
                    full_features.append(batch_features[valid_idx])
                    valid_idx += 1
                else:
                    full_features.append(None)

        return full_features

    def _start_feature_extraction(self):
        print("\033[94m Starting feature extraction...")
        image_data = pd.DataFrame()
        image_data['images_paths'] = self.image_list

        if self.use_batch_processing : 
            if self.use_gpu:
                f_data = self._get_feature_batch(self.image_list,self.image_batch_size,self.batch_size)
            else :
                f_data = self._get_feature_batch(self.image_list,self.image_batch_size,1)   # Process single image when using CPU

        else:
            f_data = self._get_feature(self.image_list)

        image_data['features'] = f_data
        image_data = image_data.dropna().reset_index(drop=True)
        image_data.to_pickle(config.image_data_with_features_pkl(self.model_name))
        print(f"\033[94m Image Meta Information Saved: [metadata-files/{self.model_name}/image_data_features.pkl]")
        return image_data

    def _start_indexing(self, image_data,batch_size=256):
        print("\033[94m Starting FAISS indexing...")
        self.image_data = image_data

        try:
            first_vector = image_data['features'][0]
            d = len(first_vector)
            self.d = d
            print(f"\033[96m Vector dimension (d): {d}")
        except Exception as e:
            print(f"\033[91m Error determining vector dimension: {e}")
            return

        try:
            features_matrix = np.vstack(image_data['features'].values).astype(np.float32)
            print(f"\033[96m Feature matrix shape: {features_matrix.shape}")
            if self.use_gpu :
                self._clear_gpu_cache()
        except Exception as e:
            print(f"\033[91m Error stacking features: {e}")
            return
        
        try:
            if faiss.get_num_gpus() > 0 and self.use_gpu:
                print("\033[92m Using GPU for FAISS indexing...")
                res = faiss.StandardGpuResources()
                index_cpu = faiss.IndexFlatL2(d)
                # index_cpu = faiss.IndexFlatIP(d)
                index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                print("\033[93m Using CPU for FAISS indexing...")
                index = faiss.IndexFlatL2(d)
                # index = faiss.IndexFlatIP(d)
            
            
            num_vectors = features_matrix.shape[0]
            total_added = 0
            print(f"\033[94m Number of feature vectors to Index: {num_vectors}")
            start_time = time.time()
            # Calculate total steps by ceil division
            num_batches = (num_vectors + batch_size - 1) // batch_size
            print(f"\033[96m Total batches to index: {num_batches}")

            with tqdm(total=num_vectors, desc="Indexing features", ncols=100) as pbar:
                for i in range(0, num_vectors, batch_size):
                    batch = features_matrix[i:i + batch_size]
                    batch_start = time.time()
                    index.add(batch)
                    batch_elapsed = time.time() - batch_start
                    batch_speed = batch.shape[0] / batch_elapsed if batch_elapsed > 0 else float('inf')

                    total_added += batch.shape[0]
                    pbar.update(batch.shape[0])
                    pbar.set_postfix(speed=f"{batch_speed:.2f} vec/s")

            total_elapsed = time.time() - start_time
            overall_speed = total_added / total_elapsed if total_elapsed > 0 else float('inf')

            print(f"\033[92m Indexed {total_added} vectors in {total_elapsed:.2f}s "
                f"({overall_speed:.2f} vectors/sec)")

            index_path = config.image_features_vectors_idx(self.model_name)
            if faiss.get_num_gpus() > 0:
                faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
            else:
                faiss.write_index(index, index_path)

            # Assign the index to the class variable for later use
            self.index = index
            print(f"\033[94m FAISS index saved to: {index_path}")

        except Exception as e:
            print(f"\033[91m Error during FAISS indexing: {e}")

        # print(f"\033[94m Saved The Indexed File: [metadata-files/{self.model_name}/image_features_vectors.idx]")

    def run_index(self, force_reextract: str = None):
        """
        Indexes the images in the image_list and creates an index file for fast similarity search.

        Parameters:
        -----------
        force_reextract : str, optional
            If set to 'yes', forces feature extraction and indexing again.
            If set to 'no', skips extraction and loads existing metadata/index.
            If None, asks the user via input().
        """
        metadata_dir = f'metadata-files/{self.model_name}'

        if not os.path.exists(metadata_dir) or len(os.listdir(metadata_dir)) == 0:
            data = self._start_feature_extraction()
            if self.use_gpu :
                self._clear_gpu_cache()
            self._start_indexing(data)
        else:
            if force_reextract is None:
                print("\033[91m Metadata and Features are already present, Do you want Extract Again? Enter yes or no")
                flag = input().strip().lower()
            else:
                flag = force_reextract.strip().lower()

            if flag == 'yes':
                data = self._start_feature_extraction()
                if self.use_gpu :
                    self._clear_gpu_cache()
                self._start_indexing(data)
            else:
                print("\033[93m Meta data already Present, Please Apply Search!")
                print(os.listdir(metadata_dir))

        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        self.f = len(self.image_data['features'][0])

    def add_images_to_index(self, new_image_paths: list):
        """
        Adds new images to the existing index.

        Parameters:
        -----------
        new_image_paths : list
            A list of paths to the new images to be added to the index.
        """
        # Load existing metadata and index
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        index = faiss.read_index(config.image_features_vectors_idx(self.model_name))

        for new_image_path in tqdm(new_image_paths):
            # Extract features from the new image
            try:
                img = Image.open(new_image_path)
                feature = self._extract(img)
            except Exception as e:
                print(f"\033[91m Error extracting features from the new image: {e}")
                continue

            # Add the new image to the metadata
            new_metadata = pd.DataFrame({"images_paths": [new_image_path], "features": [feature]})
            #self.image_data = self.image_data.append(new_metadata, ignore_index=True)
            self.image_data  =pd.concat([self.image_data, new_metadata], axis=0, ignore_index=True)

            # Add the new image to the index
            index.add(np.array([feature], dtype=np.float32))

        # Save the updated metadata and index
        self.image_data.to_pickle(config.image_data_with_features_pkl(self.model_name))
        faiss.write_index(index, config.image_features_vectors_idx(self.model_name))
        if self.use_gpu :
            self._clear_gpu_cache()
        print(f"\033[92m New images added to the index: {len(new_image_paths)}")

    def _search_by_vector(self, v, n: int):
        self.v = v
        self.n = n

        index_cpu = faiss.read_index(config.image_features_vectors_idx(self.model_name))

        if faiss.get_num_gpus() and self.use_gpu > 0:
            print("\033[92m Using GPU for FAISS search")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            print("\033[93m Using CPU for FAISS search")
            index = index_cpu

        D, I = index.search(np.array([self.v], dtype=np.float32), self.n)
        return dict(zip(I[0], self.image_data.iloc[I[0]]['images_paths'].to_list()))

 
######################### Custom functions ########################################################

        
    def _search_by_vector_curator(self, v, threshold: float = 0.7):
        """
        Search and log distances for images using the given threshold.
        """
        index_cpu = faiss.read_index(config.image_features_vectors_idx(self.model_name))
        if faiss.get_num_gpus() and self.use_gpu > 0:
            print("\033[92m Using GPU for FAISS search")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            print("\033[93m Using CPU for FAISS search")
            index = index_cpu
            
        total_vectors = index.ntotal
        if total_vectors == 0:
            print("[WARNING] FAISS index is empty.")
            return None

        D, I = index.search(np.array([v], dtype=np.float32), total_vectors)

        print(f"Total images to classify: {len(D[0])}")  # Log the number of images
        print(f"Distances for all images in the index:")

        for idx, dist in enumerate(D[0]):
            if dist >= threshold:
                print(f"Distance for image {I[0][idx]}: {dist:.4f} → Not similar")  # Indicate not similar

        top_distance = D[0][0]
        top_index = I[0][0]

        print(f"[INFO] Top FAISS distance: {top_distance}")
        print(f"[INFO] Top FAISS index: {top_index}")

        if top_distance < threshold: 
            return {
                "index": top_index,
                "distance": top_distance
            }
        else:
            return None  
   
    def get_similar_images_curator(self, image_path: str, threshold: float = 0.7):
        """
        Consistent threshold across the functions.
        """
        temp_folder = "/media/shubham/ssd_hub1/neha_work/visual_env/env/DeepImageSearch/temp"
        image_name = os.path.basename(image_path)
        print(f"\nChecking image: {image_name}")

        # Step 1: Extract the vector
        print("Extracting feature vector...")
        query_vector = self._get_query_vector(image_path)

        # Step 2: Search using FAISS
        print("Searching in FAISS index...")
        result = self._search_by_vector_curator(query_vector, threshold)  # Use the same threshold here

        if result is None:
            print("No match found in FAISS index. Skipping similarity check.")
            return False

        distance = result['distance']
        print(f"Similarity distance from closest match: {distance:.4f}")

        # Step 3: Compare and act
        if distance > threshold:
            print(f"Distance below threshold ({threshold}) → Not similar.")
            os.makedirs(temp_folder, exist_ok=True)
            target_path = os.path.join(temp_folder, image_name)
            shutil.move(image_path, target_path)

            # Print the image moved to the temp folder along with its distance
            print(f"Image moved to TEMP folder: {target_path}")
            print(f"Distance of the image: {distance:.4f}")

            try:
                image = Image.open(target_path)
                image.show(title=f"Moved Image: {image_name} | Distance: {distance:.4f}")
            except Exception as e:
                print(f"Could not open the image: {e}")

            return False
        else:
            print(f"Distance meets threshold ({threshold}) → Image is similar. No move needed.")
            return True


    # def find_similarity_images(self, unannotated_image_folder,threshold):


    def annotate_folder_parallel(self, unannotated_image_folder, output_csv, threshold):
        unannotated_image_list = sorted([
            os.path.join(unannotated_image_folder, f)
            for f in os.listdir(unannotated_image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not unannotated_image_list:
            raise ValueError("No valid images found in the specified folder.")
        
        # Extract features of all images in unannotated folder 
        print(f"Extracting features from {len(unannotated_image_list)} unannotated images...")
        
        unannotated_dataset_features = self._get_feature_batch(unannotated_image_list,1000,32)
        unannotated_features = np.vstack(unannotated_dataset_features).astype('float32')  # (N, D)
        # Confirm normalization of unannotated features
        norms = np.linalg.norm(unannotated_features, axis=1)
        print(f"Unannotated Feature norms ( Should be close to 1.0): min = {norms.min():.4f}, max = {norms.max():.4f}")
        
        self._clear_gpu_cache()
        
        # Step 3: Ensure GPU index is loaded
        if not hasattr(self, 'index') or self.index is None:
            print("Loading FAISS index...")
            index_path = config.image_features_vectors_idx(self.model_name)
            index_cpu = faiss.read_index(index_path)

            if faiss.get_num_gpus() > 0 and self.use_gpu:
                print("\033[92m Using GPU for FAISS search")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                print("\033[93m Using CPU for FAISS search")
                self.index = index_cpu
            #self.index = index_cpu
        # print("Index dimension:", self.index.d)  # From FAISS index
        # print("Unannotated feature dimension:", unannotated_features.shape[1])

        # Step 4: Perform batch FAISS search
        top_k = self.index.ntotal
        D, I = self.index.search(unannotated_features, top_k)
        self.plot_distance_distribution(D,"unnanotated")
        
        print("\033[92m FAISS search completed.")
        print("Min Distance:", D.min(), "Max Distance:", D.max())
        # Print shape and sample values
        print("Shape of D (Distances):", D.shape)
        print("Shape of I (Indices):", I.shape)
        
        print("\nSample matches (below threshold):")
        sample_count = 0
        max_samples_to_print = 10  # Limit the number of prints
        # Step 5: Load image paths for indexed (reference) images
        if not hasattr(self, 'image_list'):
            raise RuntimeError("Missing 'image_list' list for indexed dataset.")

        # # Step 6: Collect threshold-based matches
        results = []
        for i, distances in enumerate(D):
            query_path = unannotated_image_list[i]
            for j, dist in enumerate(distances):
                if dist <= threshold:
                    reference_path = self.image_list[I[i][j]]
                    results.append([
                        os.path.basename(query_path),
                        os.path.basename(reference_path),
                        float(dist)
                    ])
                    if sample_count < max_samples_to_print:
                        print(f"Query: {os.path.basename(query_path)} ↔ Reference: {os.path.basename(reference_path)} | Distance: {dist:.4f}")
                        sample_count += 1

        # Step 7: Save results to CSV
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Query_Image", "Reference_Image", "Distance"])
            writer.writerows(results)

        print(f"\033[92m Matching complete. {len(results)} pairs saved to {output_csv}")


    def annotate_folder_sequential(self, unannotated_image_folder, output_csv, threshold):
        unannotated_image_list = sorted([
            os.path.join(unannotated_image_folder, f)
            for f in os.listdir(unannotated_image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not unannotated_image_list:
            raise ValueError("No valid images found in the specified folder.")
        
        # Extract features of all images in unannotated folder 
        print(f"Extracting features from {len(unannotated_image_list)} unannotated images...")
        
        unannotated_dataset_features = self._get_feature_batch(unannotated_image_list,1000,32)
        unannotated_features = np.vstack(unannotated_dataset_features).astype('float32')  # (N, D)
        # Confirm normalization of unannotated features
        norms = np.linalg.norm(unannotated_features, axis=1)
        print(f"Unannotated Feature norms ( Should be close to 1.0): min = {norms.min():.4f}, max = {norms.max():.4f}")
        
        self._clear_gpu_cache()
        
        # Step 3: Ensure GPU index is loaded
        if not hasattr(self, 'index') or self.index is None:
            print("Loading FAISS index...")
            index_path = config.image_features_vectors_idx(self.model_name)
            index_cpu = faiss.read_index(index_path)

            if faiss.get_num_gpus() > 0 and self.use_gpu:
                print("\033[92m Using GPU for FAISS search")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                print("\033[93m Using CPU for FAISS search")
                self.index = index_cpu

        results = []
        max_samples_to_print = 10
        sample_count = 0
        
        
        print("\033[96m[INFO] Starting per-image FAISS search...")
        
        for idx in tqdm(range(len(unannotated_features)), desc="FAISS Search", unit="img"):
            query_feat = unannotated_features[idx]
            query_path = unannotated_image_list[idx]

            # Normalize (if needed)
            # query_feat = query_feat / np.linalg.norm(query_feat)
            query_feat = query_feat.astype('float32').reshape(1, -1)

            # Search FAISS index
            D, I = self.index.search(query_feat, self.index.ntotal)

            for j, dist in enumerate(D[0]):
                if dist <= threshold:
                    reference_path = self.image_list[I[0][j]]
                    results.append([
                        os.path.basename(query_path),
                        os.path.basename(reference_path),
                        float(dist)
                    ])
                    if sample_count < max_samples_to_print:
                        print(f"Query: {os.path.basename(query_path)} ↔ Reference: {os.path.basename(reference_path)} | Distance: {dist:.4f}")
                        sample_count += 1

        print(f"\033[92m Matching complete. {len(results)} pairs saved to {output_csv}")

        # Save results to CSV
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Query_Image", "Reference_Image", "Distance"])
            writer.writerows(results)

    
########################################################################

    def _get_query_vector(self, image_path: str):
        self.image_path = image_path
        img = Image.open(self.image_path)
        query_vector = self._extract(img)
        return query_vector

    def plot_similar_images(self, image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, (224, 224), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title('Input Image', fontsize=18)
        plt.imshow(input_img_resized)
        plt.show()

        query_vector = self._get_query_vector(image_path)
        img_list = list(self._search_by_vector(query_vector, number_of_images).values())
        if self.use_gpu :    
            self._clear_gpu_cache()

        grid_size = math.ceil(math.sqrt(number_of_images))
        axes = []
        fig = plt.figure(figsize=(20, 15))
        for a in range(number_of_images):
            axes.append(fig.add_subplot(grid_size, grid_size, a + 1))
            plt.axis('off')
            img = Image.open(img_list[a])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle('Similar Result Found', fontsize=22)
        plt.show(fig)

    def get_similar_images(self, image_path: str, number_of_images: int = 10):
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self._get_query_vector(self.image_path)
        img_dict = self._search_by_vector(query_vector, self.number_of_images)
        if self.use_gpu :    
            self._clear_gpu_cache()
        return img_dict
    
    def get_image_metadata_file(self):
        """
        Returns the metadata file containing information about the indexed images.

        Returns:
        --------
        DataFrame
            The Panda DataFrame of the metadata file.
        """
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        return self.image_data


    def plot_distance_distribution(self,D,image_name):
        # Flatten all distances into one array
        all_distances = D.flatten()

        # Plot histogram
        plt.figure(figsize=(10, 5))
        plt.hist(all_distances, bins=100, color='skyblue', edgecolor='black')
        plt.title("Histogram of FAISS L2 Distances")
        plt.xlabel("L2 Distance")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/home/wi/work/annotation_transfer_tool_gpu/'+image_name+'_distance_distribution.png')
