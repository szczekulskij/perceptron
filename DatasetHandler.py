import numpy as np
from collections import Counter
from random import shuffle
from Constant import Constant

class DatasetHandler:
    def __init__(self, path=Constant.DATASET_DIR):
        self.path = path
        self.raw_dataset = None
        self.feature_dataset = None
        self.label_dataset = None

    # Load the dataset into an array of numpy arrays. Stores as instance of this object
    def load_dataset(self, fileName: str):
        openedFile = open(f'{self.path}/{fileName}', "r")
        fileLines = openedFile.readlines()
        dataset = []

        for line in fileLines:
            line = line.strip(Constant.NEWLINE)
            split_data = line.split(Constant.DATASET_DELIMETER)
            coverted_class = self.convert_class_to_int(split_data)
            numpy_array = np.array(coverted_class, np.float)
            dataset.append(numpy_array)
        self.raw_dataset = dataset

    # Converts the string label of a given dataset entry to an integer
    def convert_class_to_int(self, row):
        if row[-1] == Constant.CLASSES[1]: row[-1] = 1
        elif row[-1] == Constant.CLASSES[2]: row[-1] = 2
        elif row[-1] == Constant.CLASSES[3]: row[-1] = 3
        return row

    # Make dataset includes all corresponding label as integer
    # The randomise set to False so the data order will be the same as the dataset file
    def extract_all_classes_from_dataset(self, fileName: str, randomise=False):
        self.load_dataset(fileName)
        shuffle(self.raw_dataset) if randomise else None
        feature_dataset, label_dataset = [], []

        for row in self.raw_dataset:
            label_dataset.append(row[-1])
            feature_dataset.append(row[:-1])

        self.feature_dataset, self.label_dataset = feature_dataset, label_dataset
        return self

    # Make the dataset only consist of two selected classes
    # The two classes label are either 1 or -1
    def extract_two_classes_from_dataset(self, fileName: str, classOne, classTwo, randomise=False):
        self.load_dataset(fileName)
        shuffle(self.raw_dataset) if randomise else None
        feature_dataset, label_dataset = [], []

        for row in self.raw_dataset:
            if row[-1] == classOne:
                label_dataset.append(1)
                feature_dataset.append(row[:-1])
            elif row[-1] == classTwo:
                label_dataset.append(-1)
                feature_dataset.append(row[:-1])

        self.feature_dataset, self.label_dataset = feature_dataset, label_dataset
        return self

    # Make one class positive +1 and the rest negative -1
    def make_one_class_positive(self, targetClass):
        for i in range(len(self.label_dataset)):
            if self.label_dataset[i] == targetClass:
                self.label_dataset[i] = 1
            else:
                self.label_dataset[i] = -1
        return self

if __name__ == "__main__":
    # Loads the test dataset for class 1 and class 2 in order 
    test1 = DatasetHandler()
    test1.extract_two_classes_from_dataset('test.data', 1, 2)

    loadData = DatasetHandler()
    loadData.extract_all_classes_from_dataset('test.data')

    # Loads the train and test dataset for class 2 and class 3 in randomised order
    train2 = DatasetHandler().extract_two_classes_from_dataset('train.data', 2, 3, randomise=True)
    test2 = DatasetHandler().extract_two_classes_from_dataset('test.data', 2, 3, randomise=True)

    # Loads all the test dataset for all classes. Make the only class two positive and rest negative
    testAll = DatasetHandler().extract_all_classes_from_dataset('test.data')

    # Testing if the dataset are loaded correctly
    print(len(test1.feature_dataset))
    for i in range(len(test1.feature_dataset)):
        print(f'features - {test1.feature_dataset[i]} label - {test1.label_dataset[i]}')

    # print("\nTesting all test dataset with one positive and rest negative\n")
    # print(len(testAll.feature_dataset))
    # for i in range(len(testAll.feature_dataset)):
    #     print(f'features - {testAll.feature_dataset[i]}, label - {testAll.label_dataset[i]}')