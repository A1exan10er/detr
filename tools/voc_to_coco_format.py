# Convert VOC format to COCO format
# Original dataset is in VOC format
# In the COCO format for object detection, "annotations" and "categories" are required
# Official document from COCO dataset: https://cocodataset.org/#format-data

'''
annotation{
    "id": int, 
    "image_id": int, 
    "category_id": int, 
    "segmentation": RLE or [polygon], 
    "area": float, 
    "bbox": [x,y,width,height], 
    "iscrowd": 0 or 1,
}

categories[{
    "id": int, 
    "name": str, 
    "supercategory": str,
}]
'''


import json

def read_and_convert_to_coco(input_file, output_file):
    # Initialize COCO structure
    coco_structure = {
        "info": {
            "description": "DirtNet 2019 Dataset",
            "url": "https://ieeexplore.ieee.org/document/9196559",
            "version": "1.0",
            "year": 2019,
            "contributor": "Fraunhofer IPA",
            "date_created": "2025/01/11"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define your categories
    category_mapping = {
        "dirt": 0,
        "pens": 1,
        "paper_and_notebooks": 2,
        "keys": 3,
        "usb_sticks": 4,
        "other": 5,
        "rulers": 6,
        "business_cards": 7,
        "scissors": 8,
        "tapes": 9
    }
    coco_structure["categories"] = [
        {"supercategory": "object", "id": cid, "name": cname}
        for cname, cid in category_mapping.items()
    ]

    # Process the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    image_id_map = {}  # Map to track unique images and their IDs
    annotation_id = 1

    for line in lines:
        # Parse the line
        parts = line.strip().split()
        file_name = parts[0]
        x_min = float(parts[1])
        y_min = float(parts[2])
        x_max = float(parts[3])
        y_max = float(parts[4])
        category_name = parts[5]

        # Calculate bounding box and area
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min] # According to COCO documentation: "bbox": [x,y,width,height]
        area = bbox[2] * bbox[3]

        # Get category ID
        category_id = category_mapping[category_name]

        # Assign a unique image ID if it's a new image
        if file_name not in image_id_map:
            image_id = len(image_id_map) + 1
            image_id_map[file_name] = image_id

            file_name_with_extension = f"{file_name}.png"  # File extension needs to be added

            # Add the image metadata
            coco_structure["images"].append({
                "license": 1,
                "file_name": file_name_with_extension,
                "coco_url": "", # Change this if uploading images to a server
                "height": 1024,
                "width": 1280,
                "date_captured": "",
                # "flickr_url": "", # If image from Flickr, add this
                "id": image_id
            })

        # Add the annotation
        coco_structure["annotations"].append({
            "segmentation": [],  # Add segmentation if available
            "area": area,
            "iscrowd": 0,
            "image_id": image_id_map[file_name],
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        })

        # Increment the annotation ID
        annotation_id += 1

    # Write the COCO structure to the output file
    with open(output_file, 'w') as output:
        json.dump(coco_structure, output, indent=4)

# File paths
input_file = "/home/tianyu/Documents/dirt_detection/DirtDetectionData/dataset_2019/training_synthetic/blended_floor_images/bbox_training.txt"
output_file = "bbox_training_coco_9248.json"
# input_file = "/home/tianyu/Documents/dirt_detection/DirtDetectionData/dataset_2019/training_synthetic/blended_floor_images/bbox_val.txt"
# output_file = "bbox_val_coco_9248.json"

# Convert the file to COCO format
read_and_convert_to_coco(input_file, output_file)
