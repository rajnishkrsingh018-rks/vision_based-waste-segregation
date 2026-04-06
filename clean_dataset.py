from PIL import Image
import os

def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print("Removing:", path)
                os.remove(path)

clean_folder("dataset_level1")
clean_folder("dataset_bio")
clean_folder("dataset_nonbio")