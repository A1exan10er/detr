# This Python script is used to format the bounding box labels from the original dataset to the format required by YOLOv8.
# All generated label files are stored in a folder named 'labels' in the same folder as this script.
# The original dataset format is: <image_name> <x_left> <y_top> <x_right> <y_bottom> <class_label>
# The YOLOv8 format is: <class_label_index> <x_center> <y_center> <width> <height> (all values are normalized)

# Create a folder for label files
import os
label_folder = 'labels' # The folder name for the label files
os.mkdir(label_folder) # Create a folder for label files
                       # If the folder already exists, an error will be thrown

# Open the file with the bounding box labels
with open('bbox_labels.txt', 'r') as file:
    # Read the file and store each line (image) in a list
    data = file.readlines()

    # Process each line (image) label
    for i in range(len(data)):
        data[i] = data[i].replace('\n', '')
        data[i] = data[i].replace(' ', ',')
        data[i] = data[i].split(',')
        data[i].pop() # Remove the last empty element from the list

        # Calculate the normalized center coordinates of the bounding box
        x_left = int(data[i][1])
        x_right = int(data[i][3])
        y_top = int(data[i][2])
        y_bottom = int(data[i][4])
        full_image_width = 1280.0 # The width of the image, used to normalize the coordinates, float to avoid integer division
        full_image_height = 1024.0 # The height of the image, used to normalize the coordinates
        x_center = (x_left + x_right) / 2.0 / full_image_width
        y_center = (y_top + y_bottom) / 2.0 / full_image_height

        # Calculate the width and height of the bounding box
        width = (x_right - x_left) / full_image_width
        height = (y_bottom - y_top) / full_image_height

        # Replace the bounding box coordinates with the normalized center coordinates
        data[i][1] = x_center
        data[i][2] = y_center

        # Replace the bounding box coordinates with the normalized width and height
        data[i][3] = width
        data[i][4] = height

        # List all classes and replace the class label with the index of the class
        # In total there are 10 classes: dirt, pens, paper_and_notebooks, keys, usb_sticks, other, rulers, business_cards, scissors, and tapes
        class_labels = {'dirt':0, 'pens':1, 'paper_and_notebooks':2, 'keys':3, 'usb_sticks':4, 'other':5, 'rulers':6, 'business_cards':7, 'scissors':8, 'tapes':9}
        # Replace the class label with the index of the class
        data[i][-1] = class_labels[data[i][-1]]

        # Or 2 classes: dirt and other
        # class_labels = {'dirt':0, 'other':1}

        # Write the label to the text file and all labels with the same image name will be appended to the same file
        label_path = os.path.join('./%s' % label_folder, data[i][0] + '.txt')
        with open(label_path,"a") as f:
            f.write("%s %s %s %s %s \n" % (data[i][-1], data[i][1], data[i][2], data[i][3], data[i][4])) # Format: <class_label_index> <x_center> <y_center> <width> <height>
            f.close()

    # Print the first and last line of the file to check if the format is correct
    print(data[len(data)-1])
    print('%s %s %s %s %s' % (data[len(data)-1][-1], data[len(data)-1][1], data[len(data)-1][2], data[len(data)-1][3], data[len(data)-1][4]))
    print(data[0][1])