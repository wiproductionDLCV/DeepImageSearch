import json
import sys
import boto3
import os

##
# @brief Extracts image paths from a JSON object.
#
# @param json_data The JSON data loaded from a file.
# @return A list of image path strings extracted from the "items" list in the JSON.
#
def extract_image_paths(json_data):
    image_paths = []
    for item in json_data.get("items", []):
        image_info = item.get("image", {})
        image_path = image_info.get("path")
        if image_path:
            image_paths.append(image_path)
    return image_paths


##
# @brief Downloads images from an S3 bucket using the provided image paths.
#
# @param bucket_name Name of the S3 bucket.
# @param s3_client Boto3 S3 client instance.
# @param image_paths List of image paths to download.
# @param download_dir Directory where images should be downloaded (default: "downloads").
#
def download_from_s3(bucket_name, s3_client, image_paths, download_dir="downloads"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for image_path in image_paths:
        local_path = os.path.join(download_dir, os.path.basename(image_path))
        try:
            s3_client.download_file(bucket_name, image_path, local_path)
            print(f"✅ Downloaded: {image_path} -> {local_path}")
        except Exception as e:
            print(f"❌ Failed to download {image_path}: {e}")


##
# @brief Main function to extract image paths from a JSON file and download them from an S3 bucket.
#
# @param json_file_path Path to the JSON file.
# @param bucket_name Name of the S3 bucket.
#
def main(json_file_path, bucket_name):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        image_paths = extract_image_paths(data)

        # Initialize S3 client
        s3_client = boto3.client('s3')

        print("📥 Starting download from S3...")
        download_from_s3(bucket_name, s3_client, image_paths)
        print("✅ All downloads attempted.")

    except Exception as e:
        print(f"Error: {e}")


##
# @brief Entry point of the script. Expects two command-line arguments:
#        path to the JSON file and the S3 bucket name.
#
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_and_download_images.py <path_to_json_file> <s3_bucket_name>")
    else:
        json_file = sys.argv[1]
        bucket_name = sys.argv[2]
        main(json_file, bucket_name)