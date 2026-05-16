# Part 2 – CNN Computer Vision: Surface Defect Classification

## 1. Problem overview

This project uses a Convolutional Neural Network (CNN) to classify surface images into four categories:

- **normal**
- **scratch**
- **dent**
- **stain**

Each image belongs to exactly one class, so this is a **single-label image classification** problem.

The dataset is defined in `labels.csv`, which maps each image file to a class label, for example:

- `images/normal/normal_001.png,normal`
- `images/scratch/scratch_001.png,scratch`
- `images/dent/dent_001.png,dent`
- `images/stain/stain_001.png,stain`

---

## 2. Dataset description

- **Number of classes:** 4  
- **Classes:** `normal`, `scratch`, `dent`, `stain`  
- **Images per class:** 120 each (balanced dataset)  
- **Total images:** 480  

Images are stored in subfolders:

- `images/normal/normal_XXX.png`
- `images/scratch/scratch_XXX.png`
- `images/dent/dent_XXX.png`
- `images/stain/stain_XXX.png`

The notebook:

- Loads `labels.csv`
- Counts images per class
- Displays sample images from each class
- Prints image dimensions

---

## 3. Preprocessing pipeline

Steps used in the notebook:

1. **Load labels** from `labels.csv`.
2. **Read images** from disk using OpenCV.
3. **Resize** all images to `128 × 128`.
4. **Normalize** pixel values to `[0, 1]`.
5. **Encode labels** as integers (`0–3`).
6. **Train–test split** (80% train, 20% test) with stratification.
7. **Data augmentation** on the training set:
   - Random rotation
   - Random zoom
   - Horizontal flip

---

## 4. CNN model architecture

Implemented using **TensorFlow/Keras**:

- **Input:** `(128, 128, 3)`
- **Conv2D(32, 3×3, ReLU)** → **MaxPooling2D(2×2)**
- **Conv2D(64, 3×3, ReLU)** → **MaxPooling2D(2×2)**
- **Conv2D(128, 3×3, ReLU)** → **MaxPooling2D(2×2)**
- **Flatten**
- **Dense(128, ReLU)** + Dropout(0.3)
- **Dense(4, Softmax)** (4 classes)

Compiled with:

- **Optimizer:** Adam  
- **Loss:** `sparse_categorical_crossentropy`  
- **Metrics:** `accuracy`

---

## 5. Training and evaluation

The notebook:

1. Trains the model for a configurable number of epochs (e.g., 20).
2. Tracks **training** and **validation** accuracy and loss.
3. Saves accuracy/loss curves to:

   - `results/accuracy_loss_curves.png`

4. Evaluates on the test set and prints:

   - Test loss
   - Test accuracy

5. Computes and plots a **confusion matrix**, saved as:

   - `results/confusion_matrix.png`

6. Generates **sample predictions** on test images and saves a grid of images with predicted vs true labels to:

   - `sample_predictions/prediction_outputs.png`

---

## 6. CNN concepts (simple explanation)

### What is convolution?

Convolution uses a small filter (kernel) that slides over the image and computes a weighted sum of pixel values. This helps the network detect patterns like edges, corners, textures, and shapes.

### Why is pooling used?

Pooling (e.g., max pooling):

- Reduces the spatial size of feature maps
- Keeps the most important information
- Makes the model faster and less sensitive to small shifts or noise

### Why is ReLU commonly used?

ReLU (Rectified Linear Unit):

- Sets negative values to 0 and keeps positive values
- Makes training faster
- Helps avoid the vanishing gradient problem

### Why are CNNs better than regular feed-forward networks for images?

CNNs:

- Exploit **spatial structure** in images
- Share weights across locations (filters slide over the image)
- Learn local patterns (edges → shapes → objects)

Feed-forward networks treat each pixel independently and ignore spatial relationships, making them less efficient and less accurate for image data.

---

## 7. Business use case – Manufacturing quality inspection

This CNN-based classifier can be used in **manufacturing** to automatically detect surface defects:

- **Use case:** Classify parts as `normal`, `scratch`, `dent`, or `stain`.
- **Benefits:**
  - Reduce manual inspection time and labor cost
  - Improve consistency and reduce human error
  - Trigger automatic rejection of defective parts
  - Maintain high quality standards and customer satisfaction

Example domains:

- Automotive body panels
- Consumer electronics casings
- Metal fabrication and machining

---

## 8. How to run

### 1. Create environment and install dependencies

```bash
cd part-2-cnn-computer-vision
pip install -r requirements.txt
