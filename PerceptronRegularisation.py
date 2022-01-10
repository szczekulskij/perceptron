# Task 2 - Implement a binary perceptron
# Task 3 - Use the binary perceptron to train classifiers:

import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron
from PerceptronMultiClassification import PerceptronMultiClassification
from Constant import view_dataset

class PerceptronRegularisation(Perceptron):
    def __init__(self, weight, bias):
        super().__init__(weight, bias)

    # Train the perceptron model given feature and corresponding label dataset. Model can be trained for n number of iterations
    # In our case the learning rate will be at 1 for Perceptron
    def train_perceptron_l2(self, feature_dataset, label_dataset, coefficient, num_epoch=20, randomise=True):
        weight = np.zeros(len(feature_dataset[0]))    
        bias = 0                                                  
        for _ in range(num_epoch):
            if randomise:
                feature_dataset, label_dataset = self.randomise_dataset(feature_dataset, label_dataset)
            for i, row in enumerate(feature_dataset):
                activation = np.dot(weight, row) + bias
                if label_dataset[i] * np.sign(activation) <= 0:  
                    weight = self.apply_l2_regularisation(weight, row, label_dataset[i], coefficient)
                    bias = bias + label_dataset[i]
        self.weight, self.bias = weight, bias

    # Returns the new weight after we apply the stochastic gradient update and regularisation term
    def apply_l2_regularisation(self, old_weight, row, class_value, coefficient):
        new_weight = np.multiply((1-2*coefficient), old_weight)             # W = (1-2cofficient) * W + yi * Xi
        result = np.add(new_weight, np.multiply(class_value, row))
        return result


if __name__ == "__main__":
    # Loads the training dataset for 1-vs-rest. Therefore each will have a specified class as positive and the rest negative
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(1) # class_1-vs-rest
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(2) # class_2-vs-rest
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(3) # class_3-vs-rest

    # Load the test and train dataset for classification
    test_dataset = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)
    train_dataset = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False)

    # Split the training dataset and validation set by 80% - 20%, there are 120 training data so it should be 96 - 24
    train1_training_feature, train1_training_label = train1.feature_dataset[:96], train1.feature_dataset[:96]
    train1_validation_feature, train1_validation_label = train1.feature_dataset[96:], train1.label_dataset[96:]

    REG_COEFFICIENT = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Initialise the multi-classification object
    multi_classification = PerceptronMultiClassification()

    # Training the three models with L2 coefficient 0.01
    # =======================================================================

    for coefficient in REG_COEFFICIENT:
        model1 = PerceptronRegularisation(0, 0)
        model2 = PerceptronRegularisation(0, 0)
        model3 = PerceptronRegularisation(0, 0)

        model1.train_perceptron_l2(train1.feature_dataset, train1.label_dataset, coefficient, 20)
        model2.train_perceptron_l2(train2.feature_dataset, train2.label_dataset, coefficient, 20)
        model3.train_perceptron_l2(train3.feature_dataset, train3.label_dataset, coefficient, 20)

        test_dataset_accuracy = multi_classification.compute_multiclass_prediction_accuracy(test_dataset.feature_dataset, test_dataset.label_dataset, model1, model2, model3)
        train_dataset_accuracy = multi_classification.compute_multiclass_prediction_accuracy(train_dataset.feature_dataset, train_dataset.label_dataset, model1, model2, model3)
        print(f'The result with regularisation coefficient of {coefficient}')
        print(f'Test dataset accuracy:  {test_dataset_accuracy}%')
        print(f'Train dataset accuracy: {train_dataset_accuracy}%')

    