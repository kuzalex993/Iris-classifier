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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os.path

class SIDCLASSIFIER:
    D_types = {'DT': 'Decision Tree',
               'kNN': 'k-Nearest Neighbor',
               'LR': 'Logistic Regression',
               'RFC': 'Random Forest Classifier',
               'SVM': 'Support Vector Machine'}
    clscs_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

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
        self.class_names = []

    def fetch_dataset(self) -> pd.DataFrame:
        # if dataset was already loaded once from sklearn lib to local directory(/data), we load it from local directory

        if os.path.isfile('data/iris_data.csv'):
            print('File exist')
            iris_frame = pd.read_csv('data/iris_data.csv')
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
                                                            test_size=0.30,
                                                            random_state=42)
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
            clf = LogisticRegression(solver = 'lbfgs', max_iter=100)
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

    def assess_accuracy(self, classifier, dataset: pd.DataFrame) -> pd.DataFrame:
        # build train and test sets
        X = dataset.drop(columns=['target'])
        y = dataset.target
        # split set: 75% as train set,25% as test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        mean_cross_val_score = cross_val_score(classifier, X_train, y_train, cv=5).mean()

        # Plot non-normalized confusion matrix for SVM
        titles_options = [
            ("Confusion matrix for "+self.clsType+", without normalization", None), ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                display_labels = self.class_names,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)
            image_name = 'CM for ' + self.clsType
            plt.savefig('confusion matrix/'+image_name+'.png')

        # return pandas DateFrame with model scores
        return pd.DataFrame({'classifier_type': [self.clsType],
                             'train_score': [train_score],
                             'test_score': [test_score],
                             'cross_val_score': [mean_cross_val_score]})

    def store_to_file(self, trained_model):
        filename = "models/final_model.joblib"
        # save model
        joblib.dump(trained_model, filename)


def main():
    models = ['DT', 'kNN', 'LR', 'RFC', 'SVM']
    classifiers = dict.fromkeys(models)
    scores_frame = pd.DataFrame()
    #print(classifiers)
    for model in classifiers:
        temp = SIDCLASSIFIER(model)
        data = temp.fetch_dataset()
        trained_model = temp.train(data)
        classifiers[model] = trained_model
        trained_model_score = temp.assess_accuracy(trained_model, data)
        scores_frame = pd.concat([scores_frame, trained_model_score])

    scores_frame = scores_frame.set_index('classifier_type')
    best_score = scores_frame.cross_val_score.max()
    best_models = scores_frame.loc[scores_frame.cross_val_score == best_score]

    print(scores_frame)
    print('The best model(s) based on cross-validation:')
    print('=================================')
    print(best_models)
    final_model = SIDCLASSIFIER('RFC')
    final_model.store_to_file(classifiers['RFC'])




    #iris.store_to_file()


if __name__ == '__main__':
    main()
