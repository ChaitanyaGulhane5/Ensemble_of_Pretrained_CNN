Overview:
This project involves training a deep learning model for image classification using multiple architectures like DenseNet169, InceptionV3, and Xception. The notebook includes data preprocessing, model training, evaluation, and Batch Normalization visualization.

Huggingface space link://huggingface.co/spaces/ChaitanyaGulhane5/Ensemble_of_pretrained_CNN
Dataset link: https://data.mendeley.com/datasets/22p2vcbxfk/3
 Notebook link: https://www.kaggle.com/code/chaitanyagulhane/dl-pre-final-5ec862

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

Results:
The model successfully classifies images with high accuracy.

The Batch Normalization activations are plotted for visualization.

Performance is evaluated using precision-recall and confusion matrix.

How to Use Huggingface app.py: 
Click the 'Select a Random Image and Predict' button.

The model selects an image from the dataset and predicts its disease class.

Displays confidence level and treatment suggestions.

 Upload Your Own Image

Upload a leaf image in JPG, JPEG, or PNG format.

Click 'Predict Uploaded Image' to classify the disease.

Confidence score and treatment information are displayed.

Model Prediction Pipeline:

Preprocessing: Images are resized to 256x256 and normalized.

Prediction: The image is passed through the ensemble CNN model.

Softmax Activation: Outputs probabilities for each class.

Disease Classification: The highest probability determines the predicted class.

Display Results: The app shows the disease name, confidence score, and treatment suggestions.

Install dependencies:

Ensure you have Python 3.7+ and install the required dependencies:

pip install -r requirements.txt

Run the Application:

streamlit run app.py

Contributors:

Chaitanya Gulhane (221AI015)

Gagan Deepankar (221AI019)
