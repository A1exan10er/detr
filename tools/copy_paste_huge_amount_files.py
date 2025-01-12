import os
import shutil

source_folder = '/home/tianyu/Documents/dirt_detection/DirtDetectionData/dataset_2019/training_synthetic/blended_floor_images_46240/'
destination_folder = '/media/tianyu/Disk_2/Projects/dataset_2019_synthetic_46240/train2017/'

# Initialize a counter for the number of copied files
copied_files_count = 0

# Loop through the range and form the filenames
for i in range(16):  # 000000 to 000015 (16 files)
    for j in range(61):  # 000000 to 000060 (61 files)
        for r in range(10):  # r0 to r9 (10 variants)
            for f in range(4):  # f0 to f3 (4 variants)
                filename = f"floor_{i:06d}_img_{j:06d}_r{r}_f{f}.png"
                source_path = os.path.join(source_folder, filename)
                
                if os.path.exists(source_path):  # Check if the file exists
                    shutil.copy(source_path, destination_folder)
                    copied_files_count += 1  # Increment the counter
                else:
                    print(f"File not found: {source_path}")

# Print the total number of copied files
print(f"Total number of copied files: {copied_files_count}")
