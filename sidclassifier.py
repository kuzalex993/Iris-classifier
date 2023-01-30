# SID DDA
import string
from sklearn.datasets import load_iris
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

    def fetch_dataset(self):
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
        print(iris_frame)

    def train(self):
        """
        train the classifier on the dataset
        """
        pass

    def assess_accuracy(self):
        """
        assess the accuracy of a trained classifier
        """
        pass

    def store_to_file(self):
        """
        store trained classifier to file
        """
        pass


def main():
    """
    main steps
    """

    iris = SIDCLASSIFIER('DT')
    iris.fetch_dataset()

    #iris.train()

    #iris.assess_accuracy()

    #iris.store_to_file()


if __name__ == '__main__':
    main()
