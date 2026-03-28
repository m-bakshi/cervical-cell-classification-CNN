# A CNN framework for early cancer detection

A CNN framework to classify cervical cells from pap smear images (5 classes) for early cancer detection and prognosis. This entailed implementing transfer learning using ResNet50 on SIPaKMed datase. The goal is to assist pathologists in early cervical cancer detection by automatically identifying 5 types of cervical cells from microscopic images.

## Dataset
The model has been trained on SIPaKMeD dataset which consists of 966 cluster cell images of Pap smear slides which are manually cropped from 4049 images of isolated cells. The dataset has 5 cervical cell categories - 1) Superficial-Intermediate cells 2) Parabasal cells 3) Koilocytotic cells 4) Dysketarotic cells 5) Metaplastic cells

## Project workflow

**1. Dataset division -** Built a `DatasetDivision` class that:
- Loops over each class folder in the SIPakMed dataset
- Picks only `.bmp` files using glob, `.dat` files ignored
- Splits each class independently 60% train, 20% val, 20% test using `train_test_split` with `random_state=42`
- Second split: `test_size=0.25` on the 80% remainder gives exactly 20% val (0.25 × 0.8 = 0.20)
- Copies files into structured output folders using `shutil.copy`

*splitting is done class by class, hence every class gets exactly 60/20/20*

**2. Data augmentation -** (Augmentation was applied **only on training set**, val and test sets remained original to represent real unseen data.) Built a 'DataAugmentation' class using Albumentations that:
- Reads each training image with OpenCV
- Converts BGR into RGB since ResNet50 expects RGB (OpenCV by default reads BGR)
- Generates 5 augmented copies per image and copies the original into the output folder
- Result: training set becomes ~6× larger
  
**Augmentations applied:**
- HorizontalFlip, VerticalFlip (p=0.5 each)
- RandomRotate90 (p=0.5)
- Rotate up to 30° with reflect border (p=0.6)
- GaussianBlur or MedianBlur - one of (p=0.3)
- GaussNoise or ISONoise - one of (p=0.3)
- RandomBrightnessContrast (p=0.5)
- HueSaturationValue (p=0.4)
- CLAHE clip_limit=3.0, tile_grid=(8×8) (p=0.3)
- ElasticTransform alpha=30, sigma=5 (p=0.2)
- GridDistortion num_steps=5 (p=0.2)

When saving augmented images back to disk, RGB was converted into BGR again as OpenCV's `imwrite` expects BGR.

**3. Model training -**
Implemented transfer learning by using ResNet50 as the base model, which is pretrained on ImageNet. Removed ResNet's original classification head (top layer) and added a custom one. Dropout and L2 regularisation were added to prevent overfitting.

ResNet50 Backbone (pretrained, frozen initially)
\
        ↓
       
GlobalAveragePooling2D
\
        ↓
        
Dense 512 + BatchNorm + Dropout 0.4  (L2 regularisation)
\
        ↓
        
Dense 256 + BatchNorm + Dropout 0.3  (L2 regularisation)
\
        ↓
        
Softmax → 5 classes

<img width="1920" height="1080" alt="CERVICAL_CELL_CNN" src="https://github.com/user-attachments/assets/8328b487-93e7-430b-9b6e-efb087b69346" />

**Two Phase Training:**
\
Phase 1 - Warm up the head (10 epochs)
- Froze the entire ResNet50 backbone. Only trained the new classifier head (new Dense layers).
- Learning rate : 0.001. Imp because the classifier head starts with random weights. If we immediately touch the backbone with a random head on top, the pretrained features get destroyed.
- After 10 epochs, val accuracy reached 89.18%.

Phase 2 - Fine-tune last conv block (60 epochs)
- Unfroze ResNet50 layers from 143 onwards (last convolutional block). (ResNet50 has 175 layers total. Layers 0-142 detect universal features like edges and textures, hence no changes needed.)
- Kept BatchNorm layers (earlier layers) frozen because they detect universal features like edges and textures. Imp for ResNet50's stability as well.
- Learning rate dropped to 0.0001 which is ten times smaller, to gently nudge pretrained weights towards cervical cells without catastrophic forgetting.
- Used ReduceLROnPlateau - automatically halved LR when val_loss stopped improving
- Used EarlyStopping with restore_best_weights=True
- Result: 96.92% test accuracy

Class Weights
SIPakMed has unequal class sizes: Superficial-Intermediate has 90 images, Koilocytotic only 38. I used compute_class_weight('balanced') from sklearn to handle the imbalance and prevented the model from ignoring minority classes.

**4. Model evaluation -**
- Evaluated on the unseen test set (196 images).
**Working:**
- Loads the best saved model from `.h5`
- Test generator uses same `preprocess_input` as training 
- `shuffle=False` - imp for correct label alignment between `y_true` and `y_pred`
- Model processes all 196 images in 13 batches
- Generates full classification report using sklearn
- Plots and saves confusion matrix using seaborn heatmap

5. Metrics

<img width="1920" height="1080" alt="CERVICAL_CELL_CNN (1)" src="https://github.com/user-attachments/assets/44ec7527-1b71-4854-a23a-4a6363c64bf6" />
