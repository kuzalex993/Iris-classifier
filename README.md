# Machine Learning 4 Construction
In this project we build classifier for [Iris Plants dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)

#### Next steps have been done:
1. Data exploration
2. Training several models and choise the best one
3. Development of API for new examples classification which uses trained model fom Step 2

### Data exploration
This step has been done by Jupyter notebook. You can find the file with code and its PDF-version in the folder: [data exploration](https://github.com/kuzalex993/Iris-classifier/tree/master/data%20exploration).<br />
In a nutshell the data is well prepared: no NAs and Null values. It is well balanced. But some of the featurs have high correlation.

### Training models
I used 5 models to be trained:
- Decisoin tree
- Random Forest classifier
- Logistic regression classifier
- Method k-Nearest Neighbors
- Support vector machine

For initial trainig, data set has been split on training set and test set as 70:30.<br />
Then, as way to avoid overfitting I used crossvalidation.<br />
All models have been trained on the same data sets. kNN model schowed best result on crossvalidation.
|                 |train_score | test_score | cross_val_score|
|-----------------|------------|------------|----------------|
|*classifier_type*|            |            |                |
|DT               |   1.000000 |   0.973684 |        0.966667|
|kNN              |   0.982143 |   1.000000 |        0.980000|
|LR               |   0.973214 |   0.973684 |        0.973333|
|RFC              |   1.000000 |   0.973684 |        0.960000|
|SVM              |   0.982143 |   0.973684 |        0.966667|

The cofusion matrix you can see in folder [confusion matrix](https://github.com/kuzalex993/Iris-classifier/tree/master/confusion%20matrix).<br />
kNN model was stored as final_model on the folder [models](https://github.com/kuzalex993/Iris-classifier/tree/master/models).<br />

#### Development of API
By using fastAPI I developed simple web-interface to get user ability to use trained model.<br />
Templates for web pages are stored in forlder [templates](https://github.com/kuzalex993/Iris-classifier/tree/master/templates).<br />
In *api.py* you can start local host. By the link http://127.0.0.1:8000/ you will open web-interface where you can enter features and get class for the entered data.

#### Description of other folders
Folder *data visualization* contains plots of initial data visualization<br />
Folder *data* contains Iris dataset in .csv file


#### Answers on the questions:

- *How would you assess the quality of your source code?* - Code is not perfect, but important think is it solves our task. There are some room for improvement such as better way for class initialization, upgrading train function for models gyper parameters tuning.<br />
- *How would you ship the trained ML model to the customer?* - I think the best way to transfer model to the customer is to provide them already trained model packed into userfriendly application which they can use for its intended purpose.<br />
- *Two week after shipping your product your customer calls you and complains about low accuracy of your product. How would you react?* - I will try to det more information from the customer. I will ask exact examples which failed. Ofcourse, if it is possible. Then I will back to the data exploration step adding to dataset new examples. I can use enother models or I can try to extend dataset, e.g. by artificail way.




