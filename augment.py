

# `augment.py` - OFFLINE DATA GENERATION
# ------------------------------------------------

import os
import cv2
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from utils import seeding  
from utils import create_dir  



def load_data(path):
    """ Load Data function """
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))
    
    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))
    
    return (train_x, train_y), (test_x, test_y)



def augment_data(images, masks, save_path, augment=True):
    """ Augment Data function """
    
    size = (512, 512)
    
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # 1. Extract name
        name = os.path.basename(x).split('.')[0]

        # 2. Read Image + Mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        
        # 3. Augmentation 
        if augment == True:
            
            # [Aug 1]======================== 
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']
            
            # [Aug 2]======================== 
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            # [Aug 3]======================== 
            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']
            
            
            # Combine Original + 3 Augmentations
            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            # No augmentation for test data
            X = [x]
            Y = [y]
            
        # 4. Save Augmented Data
        
        index = 0
        for i, m in zip(X, Y):
            
            i = cv2.resize(i, size, interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)

            # Create Filenames
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name =  f"{name}_{index}.png"
            
            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask", tmp_mask_name)
            
            # Save
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            
            index += 1






if __name__ == "__main__":

    seeding(42)
    data_path = "D:/nada/______Projects/Retina-Blood-Vessel-Segmentation/DRIVE/"
    
    # Load Original Data
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Original Train: {len(train_x)} - {len(train_y)}")
    print(f"Original Test : {len(test_x)} - {len(test_y)}")





    # Define New Save Path
    save_path = "new_data" 

    # Create Directories
    create_dir(os.path.join(save_path, "train/image/"))
    create_dir(os.path.join(save_path, "train/mask/"))
    create_dir(os.path.join(save_path, "test/image/"))
    create_dir(os.path.join(save_path, "test/mask/"))





    # Run Augmentation
    print(f"Images will be saved in: {os.path.abspath(save_path)}") 



    # Training Data Augmentation
    augment_data(train_x, train_y, os.path.join(save_path, "train"), augment=True)

    # Test Data Augmentation
    augment_data(test_x, test_y, os.path.join(save_path, "test"), augment=False)


    
    print("Augmentaion is Done! Data is ready for training.")