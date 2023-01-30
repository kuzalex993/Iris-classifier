# SID DDA
import string
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os.path

class SIDCLASSIFIER:
    D_types = {'DT': 'Decision Tree',
               'kNN': 'k-Nearest Neighbor',
               'LR': 'Logistic Regression',
               'RFC': 'Random Forest Classifier',
               'SVM': 'Support Vector Machine'}

    def __init__(self, clsType: string):
        # check the entered type of classifier
        if clsType in self.D_types: # if it is correct -> show message about successful attempt
            self.clsType = clsType
            print('Object for ' + self.D_types[clsType] + ' model has been created')
        else: # if not -> show message about the selection.
            print('Please, specify type of classifier to be trained from the list')
            print('Type' + ' ----- ' + 'Description')
            # print('------------------------')
            for key in self.D_types:
                print(key + ' ----- '+ self.D_types[key])

    def fetch_dataset(self) -> pd.DataFrame:
        # if dataset was already loaded once from sklearn lib to local directory(/data), we load it from local directory

        if os.path.isfile('/data/iris_data.csv'):
            print('File exist')
            iris_frame = pd.read_csv('/data/iris_data.csv')
        else:
            print('File not exist and datset will be loaded from sklearn library')
            # load Iris data from sklearn
            iris_data = load_iris()
            print("Data set 'Iris' is loaded successful.")

            # convert data to Pandas DataFrame
            iris_frame = pd.DataFrame(iris_data.data)

            # Naming columns
            iris_frame.columns = iris_data.feature_names

            # Add Target column to DataFrame:
            iris_frame['target'] = iris_data.target
            print("Data set was converted to Pandas DataFrame")

            # Save DataFrame on local machine
            iris_frame.to_csv('data/iris_data.csv')
            print("Data set was saved to file '/data/iris_data.csv'")
        #print(iris_frame)
        return iris_frame

    def train(self, dataset: pd.DataFrame):# -> pd.DataFrame:
        # build train and test sets
        X = dataset.drop(columns=['target'])
        y = dataset.target
        # split set: 75% as train set,25% as test set
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            test_size=0.25,
                                                            random_state=12)
        # train SVM model without any tuning
        if self.clsType == 'SVM':
            clf = SVC()
            clf.fit(X_train, y_train)
            return clf
        elif self.clsType == 'DT': # train DT model
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            return clf
        elif self.clsType == 'LR': # train LogisticRegression model
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            return clf
        elif self.clsType == 'kNN':
            # 6 neighbors are optimal for this dataset, it was defined at previews step
            clf = KNeighborsClassifier(n_neighbors=6)
            clf.fit(X_train, y_train)
            return clf
        elif self.clsType == 'RFC':
            # 6 neighbors are optimal for this dataset, it was defined at previews step
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            return clf

    def assess_accuracy(self, clf, dataset: pd.DataFrame) -> pd.DataFrame:
        # build train and test sets
        X = dataset.drop(columns=['target'])
        y = dataset.target
        # split set: 75% as train set,25% as test set
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=12)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        # return pandas DateFrame with model scores
        return pd.DataFrame({'classifier_type': [self.clsType],
                             'train_score': [train_score],
                             'cross_val_score': [mean_cross_val_score]})

    def store_to_file(self):
        """
        store trained classifier to file
        """
        pass


def main():
    models = ['DT', 'kNN', 'LR', 'RFC', 'SVM']
    classifiers = dict.fromkeys(models)
    scores = pd.DataFrame()
    print(classifiers)
    for model in classifiers:
        temp = SIDCLASSIFIER(model)
        data = temp.fetch_dataset()
        trained_model = temp.train(data)
        classifiers[model] = trained_model



    #scores = pd.concat([scores, tmp_score])
    print(classifiers)

    #iris.assess_accuracy()

    #iris.store_to_file()


if __name__ == '__main__':
    main()
