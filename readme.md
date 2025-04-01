Overview:
This project involves training a deep learning model for image classification using multiple architectures like DenseNet169, InceptionV3, and Xception. The notebook includes data preprocessing, model training, evaluation, and Batch Normalization visualization.

NOTE: The h5 file of the ensemble model is too huge to be uploaded into the file so we are sharing the link for the kaggle notebook. Public Notebook link: https://www.kaggle.com/code/avyaya/dl-pre-final-5ec862

Dataset:
The dataset consists of images organized into different classes.

Data augmentation techniques such as horizontal flipping, shearing, and zooming have been applied.

The images are center-cropped to 256x256 pixels before being fed into the model.

Code Structure:
1. Data Preprocessing:
Cropping images to a fixed size.

Applying data augmentation.

Splitting into training and validation sets.

2. Model Definition:
Uses DenseNet169, InceptionV3, and Xception as backbone architectures.

Global Average Pooling, Dropout, and Dense layers are added.

Optimized using RMSprop optimizer with metrics like Precision, Recall.

3. Model Training:
The model is trained for multiple epochs.

Checkpointing and early stopping are used.

4. Evaluation & Metrics:
Computes confusion matrix, classification report, and ROC curve.

Uses Matthews Correlation Coefficient (MCC) for performance assessment.

5. Batch Normalization Analysis:
Extracts and plots Batch Normalization (BN) activations from different epochs.

Saves the plots for further analysis.

Results
The model successfully classifies images with high accuracy.

The Batch Normalization activations are plotted for visualization.

Performance is evaluated using precision-recall and confusion matrix.

How to Use
Install dependencies:


pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn

Run the Jupyter Notebook to preprocess data and train the model.

Check the saved BN plots in the bn_plots/ directory.

Future Improvements:
Fine-tuning model parameters for better performance.

Exploring additional augmentation techniques.

Implementing transfer learning with more architectures.

