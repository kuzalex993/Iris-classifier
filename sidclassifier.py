# SID DDA

import sklearn
import pandas as pd


class SIDCLASSIFIER:
    def __init__(self):
        """
        init the class
        """
        pass

    def fetch_dataset(self):
        """
        fetch dataset and store to file
        """
        pass

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

    iris = SIDCLASSIFIER()

    iris.fetch_dataset()

    iris.train()

    iris.assess_accuracy()

    iris.store_to_file()


if __name__ == '__main__':
    main()
