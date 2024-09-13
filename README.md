# Predictive Maintenance Machine Learning API

### PROJECT DESCRIPTION:
This project focuses on developing a "Predictive Maintenance Machine Learning API" that allows users to predict machine failures using multiple machine learning models. The API takes custom input data from a CSV file and uses pre-trained models, such as Logistic Regression, Random Forest, Support Vector Classifier, Gradient Boosting, K-Nearest Neighbors, and K-Means Clustering, to predict the likelihood of machine failure. 

The system is designed to assist industries in proactive maintenance by identifying potential breakdowns before they occur, reducing downtime and maintenance costs. The API is accessible via a simple web interface that enables users to upload data, receive predictions, and make informed decisions about machine maintenance.

**Dataset used:** https://www.kaggle.com/datasets/abdelazizsami/predictive-maintenance-dataset

### FILES USED IN THE PROJECT:
Following is the information of the files involved in this project:

1. **ai2020.csv:**

    This is the main training data achieved from the Kaggle website. It consists of 10000 instances of products. We have used 80%-20% train-test split and trained it on various ML models as mentioned above.

2. **predictive-maintenance.ipynb:**

    This is a colab notebook that goes through the dataset taken, and analyses it thoroughly. We have performed feature engineering, feature importance tests, cross validation results along with simple performance evaluation results of many machine learning models (related to classification). it is the base code for the analysis below.

3. **data-analysis.py:**

   This file aims to train the models on the data above. The models will then be saved using pickle package, with a .pkl format. It involves the function preprocess-data, then training it and testing them. Also we print the accuracies for performance evaluation.

5. **api.py:**

   This involves using the above trained models for predicting new results. We have pre-processed the new input data as we have done for the training data, and then predicted the outcomes. Various model outcomes are predicted in the json list format. We have used flask as the web API runner.

7. **index.html:**

   This contains the frontend code, for the flask api above to run on. It provides an upload option for user input as a "csv" file. Then predict button will provide the classification report below.

9. **styles.css:**

   This is for styling our HTML file above.

11. **requirements.txt:**

    It contains the python packages used in our code, with versions.


### HOW TO RUN THE PROJECT?
1. Make sure all the above files are in a single folder
2. Run the data-analytics.py file, it will create 6 models' pkl files (Logistic Regression, Random Forest, Support Vector Classifier, Gradient Boosting, K-Nearest Neighbors, and K-Means Clustering) along with another scaler.pkl for standardScaler (for standardisation of data in pre-processing). If already created, skip this step.
3. Now, run the api.py file. It will show that it is running on localhost 5000 port or any other specified port. 
4. Next, double-click and open the index.html directly, so that it opens in the browser. 
5. It will show you the upload button, click and upload a csv.
6. We have provided 2 sample CSVs (sample1.csv, sample2.csv) each one for a failure and not-failure case.
7. After uploading click the predict button to get results.


### COPYRIGHTS:
This project is done by myself and is not copied from any sources. It may have similarities with others' codes as the concepts are highly established, but I assure that my intention is not to plagiarise anyone's work. I present this project for knowledge enhancement and improvement in the field of machine learning.
