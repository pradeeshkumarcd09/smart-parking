# Smart Parking Space Detection using CNN

## Objective
Classify a parking-patch image as BUSY (occupied) or FREE (vacant).

## Dataset
CNRPark-Patches-150x150 dataset.
Original dataset repository:
https://github.com/imontoyah/CNN-parking-detection

## Model
Convolutional Neural Network (CNN), implemented with TensorFlow/Keras.
Input size: 128 x 128 x 3
Output: sigmoid probability for FREE (label 1).
Threshold: 0.5

## Test results
Test accuracy: 99.68%
Test loss: 0.0108
Test images: 1888

These results apply to the current random test split. Similar images from
the same parking locations or time periods may occur across splits, so
performance on entirely new parking locations must be validated separately.

## Files
- models/best_parking_model.keras : trained model
- results/test_metrics.json : test metrics
- results/classification_report.json : per-class metrics
- results/confusion_matrix.csv : confusion matrix
- results/confusion_matrix.png : confusion matrix plot, if available
- class_names.json : label mapping
- src/predict.py : prediction script

## Run prediction locally
Install Python dependencies:
pip install tensorflow pillow numpy

From the project folder, run:
python src/predict.py path/to/parking_image.jpg
