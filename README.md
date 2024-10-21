# Binary Classification of Heart Disease Using a Multi-Layer Perceptron (MLP) with Keras
## Objective

The purpose of this lab is to develop a binary classification model using a Multi-Layer Perceptron (MLP) with Keras to predict the likelihood of heart disease based on health indicators. The model will use a dataset of health metrics and indicators to predict the target variable HeartDiseaseorAttack, applying techniques like feature selection, normalization, and hyperparameter tuning.
## Dataset Description

The dataset contains 22 columns representing various health indicators for individuals. The key columns include:

**1- HighBP:** 

Indicator of high blood pressure

**2- BMI:** 

Body Mass Index

**3- Smoker:** 

Indicator of smoking status

**4- Diabetes:** 

Indicator of diabetes status

**5-HeartDiseaseorAttack:** 

Target variable (1 for presence of heart disease or attack, 0 otherwise)

Other columns in the dataset include various behavioral, demographic, and health-related indicators.
## Steps to Run the Code
**1. Prerequisites**

Make sure you have Python and the following libraries installed before running the code. You can install the required libraries using pip.

pip install pandas numpy scikit-learn seaborn matplotlib tensorflow keras keras-tuner

**2. Dataset**

Download the dataset and place it in the appropriate directory. In the code, the dataset is loaded from 'D:/noody/Deep learning/Lab 2/heart_disease_health_indicators.csv'. Adjust the file path if needed.

**3. Running the Code**

1- Open the provided Jupyter Notebook or create a new notebook, and copy the code into the notebook.

2- Ensure that the dataset is correctly loaded and paths are updated as per your directory structure.

3- Run the cells in sequence:

        i. Data Preprocessing: This step involves loading the dataset, handling missing values, and feature selection using VarianceThreshold and correlation matrix.
        
        ii. Model Building: The MLP is built using Keras, with hyperparameter tuning using Keras Tuner. The model includes multiple hidden layers with dropout and L2 regularization.
        
        iii. Model Compilation: Compile the model using the Adam optimizer and binary cross-entropy as the loss function.
        
        iv. Model Training and Evaluation: The model is trained with early stopping to prevent overfitting. The performance of the model is evaluated using accuracy, confusion matrix, precision, recall, F1-score, and the ROC-AUC curve.

        v. TensorBoard: Training logs can be visualized using TensorBoard. Run the following command to view the training metrics:

            tensorboard --logdir=logs/fit

4- Hyperparameter Tuning (Bonus)

The code uses Keras Tuner for hyperparameter tuning. It explores different combinations of layers, neurons, dropout rates, and regularization parameters to optimize the model's performance.

5- Model Evaluation

The evaluation metrics, including confusion matrix and ROC-AUC curve, are visualized at the end of the training process. You can monitor the training and validation losses over the epochs to see how well the model performs over time.

## Dependencies

**1- Python 3.x**

**2- Pandas**

(Data manipulation): 

pip install pandas

**3- NumPy**
 
(Numerical operations): 

pip install numpy

**4- Scikit-learn**

(ML utilities and evaluation metrics): 

pip install scikit-learn

**5- Seaborn**

(Data visualization): 

pip install seaborn

**6- Matplotlib** 

(Plotting): 

pip install matplotlib

**7- TensorFlow**

(Deep learning library): 

pip install tensorflow

**8- Keras** 

(High-level API of TensorFlow): 

pip install keras

**9-Keras Tuner** 

(For hyperparameter tuning): 

pip install keras-tuner