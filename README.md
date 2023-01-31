# Machine Learning 4 Construction
In this project we build classifier for [Iris Plants dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)

#### Next steps have been done:
1. Data exploration
2. Training several models and choise the best one
3. Development of API for new examples classification which uses trained model fom Step 2

### Data exploration
This step has been done by Jupyter notebook. You can find file and PDF-version in the folder: [data exploration](https://github.com/kuzalex993/Iris-classifier/tree/master/data%20exploration).
In a nutt shell the data is well prepared: no NAs and Null values. It is well balanced. But some of the feachers have high correlation.

### Training models
I used 5 models to be trained:
- Decisoin tree
- Random Forest classifier
- Logistic regression classifier
- Method k-Nearest Neighbors
- Supprt vector machine

For initial trainig, data set has been split on training set and test set as 70:30.
Then as way to avoid overfitting I used crossvalidation.
All models have been trained on the same data sets. kNN model schowed best result on crossvalidaion.
The cofusion matrix you can see in folder [confusion matrix](https://github.com/kuzalex993/Iris-classifier/tree/master/confusion%20matrix)
kNN model was stored as final_model on the folder [models](https://github.com/kuzalex993/Iris-classifier/tree/master/models)

#### Development of API
By using fastAPI I developed simple web-interface to get user ability to use trained model.
Templates for web pages are stored in forlder [templates](https://github.com/kuzalex993/Iris-classifier/tree/master/templates)
In *iris-api.py* you can start local host. By the link http://127.0.0.1:8000/ you will open 

#### Description of other folders
Folder *data visualization* contains plots of initial data visualization
Folder *data contains Iris dataset in .csv file


#### Answers on the questions:

How would you assess the quality of your source code?
How would you ship the trained ML model to the customer?
Two week after shipping your product your customer calls you and complains about low accuracy of your product. How would you react?




