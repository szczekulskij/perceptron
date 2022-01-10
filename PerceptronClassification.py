# Part 3 - Use the binary perceptron to train classifiers to discriminate between classes

import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron
from Constant import Constant

class PerceptronClassification:
    def activation_score(self, data_row, weight, bias) -> int:
        activation = np.dot(data_row, weight) + bias
        return np.sign(activation)
        
    # Given a test dataset with feature + label array. Returns the total accuracy for correct activation_scoreion
    def compute_prediction_accuracy(self, feature_dataset, label_dataset, bias, weight) -> int:
        correct_prediction = 0
        for i, data_row in enumerate(feature_dataset):
            activation_score = self.activation_score(data_row, weight, bias)
            if activation_score == label_dataset[i]:
                correct_prediction += 1
        return (correct_prediction / len(feature_dataset)) * 100

from Constant import view_dataset

if __name__ == "__main__":
    # Question 1 to 3
    # Loads the randomised dataset with two classes or all

     # Test and Train Dataset for class 1 and class 2
    train1 = DatasetHandler().extract_two_classes_from_dataset('train.data', 1, 2, randomise=False)
    test1 = DatasetHandler().extract_two_classes_from_dataset('test.data', 1, 2, randomise=False)

    view_dataset(train1.feature_dataset, train1.label_dataset)
    print(Constant.BREAKPOINT)
    view_dataset(test1.feature_dataset, test1.label_dataset)

    # Test and Train Dataset for class 2 and class 3
    train2 = DatasetHandler().extract_two_classes_from_dataset('train.data', 2, 3, randomise=False)
    test2 = DatasetHandler().extract_two_classes_from_dataset('test.data', 2, 3, randomise=False)

    # Test and Train Dataset between class 1 and class 3
    train3 = DatasetHandler().extract_two_classes_from_dataset('train.data', 1, 3, randomise=False)
    test3 = DatasetHandler().extract_two_classes_from_dataset('test.data', 1, 3, randomise=False)

    # Initialise perceptron algorithm with start weight of 0 and bias of 0
    m1 = Perceptron(0, 0)
    m2 = Perceptron(0, 0)
    m3 = Perceptron(0, 0)

    # Training the perceptron with 20 iterations for the three types of train dataset we have loaded
    m1.train_perceptron( train1.feature_dataset, train1.label_dataset, 20)
    m2.train_perceptron(train2.feature_dataset, train2.label_dataset, 20)
    m3.train_perceptron(train3.feature_dataset, train3.label_dataset, 20)

    # Initialise classification class to compute accuracy of test data
    activation_score = PerceptronClassification()

    # Compute the total accuracy for the three model given the train dataset
    trainAcc1 = activation_score.compute_prediction_accuracy(train1.feature_dataset, train1.label_dataset, m1.get_bias(), m1.get_weight())
    trainAcc2 = activation_score.compute_prediction_accuracy(train2.feature_dataset, train2.label_dataset, m2.get_bias(), m2.get_weight())
    trainAcc3 = activation_score.compute_prediction_accuracy(train3.feature_dataset, train3.label_dataset, m3.get_bias(), m3.get_weight())

    # Compute the total accuracy for the three model given the test dataset
    testAcc1 = activation_score.compute_prediction_accuracy(test1.feature_dataset, test1.label_dataset, m1.get_bias(), m1.get_weight())
    testAcc2 = activation_score.compute_prediction_accuracy(test2.feature_dataset, test2.label_dataset, m2.get_bias(), m2.get_weight())
    testAcc3 = activation_score.compute_prediction_accuracy(test3.feature_dataset, test3.label_dataset, m3.get_bias(), m3.get_weight())

    # print("The perceptron binary classifcation accuracy for training and testing dataset are:")

    # print(Constant.BREAKPOINT)

    # print(f"Train dataset with class 1 and 2 with accuracy of {trainAcc1}%")
    # print(f"Train dataset with class 2 and 3 with accuracy of {trainAcc2}%")
    # print(f"Train dataset with class 1 and 3 with accuracy of {trainAcc3}%")

    # print(Constant.BREAKPOINT)

    # print(f"Test dataset with class 1 and 2 with accuracy of {testAcc1}%")
    # print(f"Test dataset with class 2 and 3 with accuracy of {testAcc2}%")
    # print(f"Test dataset with class 1 and 3 with accuracy of {testAcc3}%")