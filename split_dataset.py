import os
from sklearn.model_selection import train_test_split
import glob
import shutil

class DatasetDivision:

    # root_dir = r"F:\Cervival_Cell_CNN\SIPakMed_dataset"
    # output_dir = r"F:\Cervival_Cell_CNN\SIPakMed_split_dataset"

    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir
        print("Instance of class created")

    def divide_dataset(self):
        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            for class_name in os.listdir(self.root_dir):
                os.makedirs(os.path.join(self.output_dir, split, class_name), exist_ok=True)

        # Iterate over each class folder 
        for class_name in os.listdir(self.root_dir): # loops over original dataset
            class_path = os.path.join(self.root_dir, class_name) # extracts path to a class folder like Dyskeratotic, "F:\Cervival_Cell_CNN\SIPakMed_dataset\Dyskeratotic"
            if not os.path.isdir(class_path): # ignores/skips if an item is a file (like .dat)
                continue

            # Only pick .bmp files (ignore .dat)
            class_files = glob.glob(os.path.join(class_path, "*.bmp"))
            if len(class_files) == 0:
                continue

            # Split using sklearn
            train_val, test = train_test_split(class_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_val, test_size=0.25, random_state=42)  # 0.25*0.8 = 0.2

            # Copy files into train/val/test folders
            for f in train:
                shutil.copy(f, os.path.join(self.output_dir, 'train', class_name, os.path.basename(f)))
            for f in val:
                shutil.copy(f, os.path.join(self.output_dir, 'val', class_name, os.path.basename(f)))
            for f in test:
                shutil.copy(f, os.path.join(self.output_dir, 'test', class_name, os.path.basename(f)))

        print("Dataset has been split, check train/val/test folders.")


# Split data into 

if __name__ == "__main__": # checks if python file is being run directly
    root = r"F:\Cervival_Cell_CNN\SIPakMed_dataset"          # original dataset
    output = r"F:\Cervival_Cell_CNN\SIPakMed_split_dataset"  # folder for split dataset

    divider = DatasetDivision(root, output) # creates  a divider object of class DatasetDivision
    divider.divide_dataset() # splits the dataset into train/val/test folders