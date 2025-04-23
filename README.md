## Skin Cancer MNIST: HAM10000
The HAM10000 dataset is a collection of over 10,000 dermatoscopic images of pigmented skin lesions. This diverse dataset is widely used for academic machine learning research, particularly in the field of skin cancer classification. The goal is to create a practical usage of Image Recognition AI to accurately predict benign and malignant lesions. This was achieved through the deep-learning architecture Resnet50, resulting in an accuracy rate of 89%.

## Data
The data consists of 10015 images of sizes 650x450 with the 3 channels RGB. There are 7 classes, but the data is seriously imbalanced.
* Actinic Keratoses(3.3%)
* Basal cell carcinoma(5.1%)
* Benign Keratosis(11%)
* Dermatofibroma(1.1%)
* Melanocytic nevi(66.9%)
* Melanoma(11.1%)
* Vascular skin lesions(1.4%)

## Methodologies
### Image/Data Processing
* Opencv resizing of image to 224x224 for Resnet50 along with color channel conversion to RGB.
* Normalization of images to 0-1 and calculation of the 3 channels' means and standard deviations of all images
* Usage of Pandas dataframe to inner join metadata class labels with images
### Data Augmentation
* 80-20 Training data and Validation Data split
* Oversampling on training data to get 1-1 samples in order to address class imbalance
* Pytorch Transforms to augment data with Horizontal/Vertical flips, rotations, and colorjitter
### Resnet50 Model
* Transfer learning of Resnet50 with all layers unfrozen
* Dataloader batch training of 32 samples
* CrossEntropyLoss and AdamW Optimizer with lr .0001 and built-in weight decay


