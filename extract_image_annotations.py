import json
import os

# Function to extract, sort, and count specific image information and annotations
def extract_and_sort_images_info(input_json_path, images_folder_path, output_json_path):
    # Load the original COCO JSON data
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Prepare output data structure with "info" and "licenses" only once
    output_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": [],
        "annotations": []
    }

    # Get all image filenames in the folder
    image_filenames = [f for f in os.listdir(images_folder_path) if f.endswith(('.jpg', '.png'))]

    # Iterate over each image file
    for filename in image_filenames:
        # Find the image data matching the current filename
        target_image = next((image for image in data["images"] if image["file_name"] == filename), None)

        if target_image:
            # Add the found image data to output
            output_data["images"].append(target_image)

            # Extract the "id" of the target image to filter annotations
            target_id = target_image["id"]

            # Find annotations corresponding to this image
            target_annotations = [annotation for annotation in data.get("annotations", []) if annotation["image_id"] == target_id]

            # Add the found annotations to output
            output_data["annotations"].extend(target_annotations)

    # Sort "images" by "id" and "annotations" by "image_id"
    output_data["images"] = sorted(output_data["images"], key=lambda x: x["id"])
    output_data["annotations"] = sorted(output_data["annotations"], key=lambda x: x["image_id"])

    # Calculate the total number of entries in "images" and "annotations"
    num_images = len(output_data["images"])
    num_annotations = len(output_data["annotations"])

    # Write the output data to a new JSON file
    with open(output_json_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    # Print total counts of "images" and "annotations"
    print(f"Total number of images: {num_images}")
    print(f"Total number of annotations: {num_annotations}")

# Example usage
input_json_path = '/home/tianyu/Projects/datasets/coco2017_small/annotations/person_keypoints_val2017.json'  # Path to the original COCO JSON file
images_folder_path = '/home/tianyu/Projects/datasets/coco2017_small/val2017'  # Folder containing images
output_json_path = '/home/tianyu/Projects/datasets/coco2017_small/annotations/person_keypoints_val2017_small.json'  # Desired output JSON file path

extract_and_sort_images_info(input_json_path, images_folder_path, output_json_path)
