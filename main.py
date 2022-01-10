import numpy as np
from random import shuffle

class Constant:
    CLASSES = ["class-0", "class-1", "class-2", "class-3"]

    # Regularisation coefficient
    REG_COEFFICIENT = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    DATASET_DIR = "dataset"

    DATASET_DELIMETER = ","
    
    NEWLINE = "\n"

    BREAKPOINT = "======" * 10


# This class deals with manipulating the data files
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
            numpy_array = np.array(coverted_class, float)
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


# The main Perceptron class 
class Perceptron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    # Train the perceptron model given feature and corresponding label dataset. Model can be trained for n number of iterations
    def train_perceptron(self, feature_dataset, label_dataset, num_epoch=20, randomise=True):
        weight = np.zeros(len(feature_dataset[0]))    
        bias = 0                                                  
        for _ in range(num_epoch):
            if randomise:
                feature_dataset, label_dataset = self.randomise_dataset(feature_dataset, label_dataset)
            for i, row in enumerate(feature_dataset):
                activation = np.dot(weight, row) + bias
                if label_dataset[i] * np.sign(activation) <= 0:  
                    weight = self.compute_new_weight(weight, row, label_dataset[i]) 
                    bias = bias + label_dataset[i]
        self.weight, self.bias = weight, bias

    # Compute the activation score
    def compute_new_weight(self, oldWeight, row, class_value):
        newRow = np.multiply(class_value, row)
        return np.add(oldWeight, newRow)

    def randomise_dataset(self, feature_dataset, label_dataset):
        combine_feature_and_label = list(zip(feature_dataset, label_dataset))
        shuffle(combine_feature_and_label)
        feature_dataset, label_dataset = zip(*combine_feature_and_label)
        return feature_dataset, label_dataset

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias


# The l2 version of Perceptron algorithm. This time l2 will be applied when we have to update the weight and bias
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
                    weight = self.apply_l2_regularisation(weight, row, label_dataset[i], coefficient) # This is where we will apply l2 regularisation term
                    bias = bias + label_dataset[i]
        self.weight, self.bias = weight, bias

    # Returns the new weight after we apply the stochastic gradient update and regularisation term
    def apply_l2_regularisation(self, old_weight, row, class_value, coefficient):
        new_weight = np.multiply((1-2*coefficient), old_weight)             # W = (1-2cofficient) * W + yi * Xi
        result = np.add(new_weight, np.multiply(class_value, row))
        return result


# The base classification class. It will take the unseen dataset and the Perceptron model and compute the accuracy of the prediction.
class PerceptronClassification:
    def activation_score(self, data_row, weight, bias) -> int:
        activation = np.dot(data_row, weight) + bias
        return np.sign(activation)
        
    # Given a test dataset with feature + label array. Returns the total accuracy for correct activation_score
    # Also input the Perceptron model, which is the bias and weight
    def compute_prediction_accuracy(self, feature_dataset, label_dataset, bias, weight) -> int:
        correct_prediction = 0
        for i, data_row in enumerate(feature_dataset):
            activation_score = self.activation_score(data_row, weight, bias)
            if activation_score == label_dataset[i]:
                correct_prediction += 1
        return (correct_prediction / len(feature_dataset)) * 100


# The extended classification class, which allows 1-vs-rest-classification, hence it can take multiple Perceptron model
class PerceptronMultiClassification(PerceptronClassification):
    def activation_score_raw(self, data_row, weight, bias) -> int:
        activation = np.dot(data_row, weight) + bias
        return activation

    def compute_multiclass_prediction_accuracy(self, feature_dataset, label_dataset, *trained_models):
        correct_prediction =  0
        for i, data_row in enumerate(feature_dataset):
            confidence_scores = [] # Stores the confidence score for each trained model index
            for model in trained_models:
                activation_score = self.activation_score_raw(data_row, model.get_weight(), model.get_bias())
                confidence_scores.append(activation_score)
            # The selected model will have the highest confidence score, we will select the model at this index to use for classification
            selected_model_index = np.argmax(confidence_scores) 
            prediction = self.activation_score(data_row, trained_models[selected_model_index].get_weight(), trained_models[selected_model_index].get_bias())
            # The model index corresponds to the class i.e. model with index 2 has been trained for training data class_2-vs-rest
            # Note, we do +1 because our class starts from 1
            # And if prediction is equals to 1, therefore it is indeed class 2 (correct prediction), hence it could be class 1, class 3, class 4...
            if selected_model_index+1 == int(label_dataset[i]) and int(prediction) == 1:
                correct_prediction += 1 
        return (correct_prediction / len(feature_dataset)) * 100


if __name__ == "__main__":
    print("\n")

    # ================================================================================================================
    # ================================================================================================================
    # Question 3 - Compute the accuracy for the three types of classes
    # ================================================================================================================
    # ================================================================================================================

     # Test and Train Dataset for class 1 and class 2
    train1 = DatasetHandler().extract_two_classes_from_dataset('train.data', 1, 2, randomise=False)
    test1 = DatasetHandler().extract_two_classes_from_dataset('test.data', 1, 2, randomise=False)

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
    classify = PerceptronClassification()

    # Compute the total accuracy for the three model given the train dataset
    trainAcc1 = classify.compute_prediction_accuracy(train1.feature_dataset, train1.label_dataset, m1.get_bias(), m1.get_weight())
    trainAcc2 = classify.compute_prediction_accuracy(train2.feature_dataset, train2.label_dataset, m2.get_bias(), m2.get_weight())
    trainAcc3 = classify.compute_prediction_accuracy(train3.feature_dataset, train3.label_dataset, m3.get_bias(), m3.get_weight())

    # Compute the total accuracy for the three model given the test dataset
    testAcc1 = classify.compute_prediction_accuracy(test1.feature_dataset, test1.label_dataset, m1.get_bias(), m1.get_weight())
    testAcc2 = classify.compute_prediction_accuracy(test2.feature_dataset, test2.label_dataset, m2.get_bias(), m2.get_weight())
    testAcc3 = classify.compute_prediction_accuracy(test3.feature_dataset, test3.label_dataset, m3.get_bias(), m3.get_weight())
    
    print("Question 3 - Report the train and test classification accuracies:")
    print("The training dataset were randomised during each epoch.")
    print(Constant.BREAKPOINT)
    print("The result for class 1 and class 2 prediction: ")
    print(f'Test dataset: {testAcc1}%')
    print(f'Train dataset: {trainAcc1}%')
    print("The result for class 2 and class 3 prediction: ")
    print(f'Test dataset: {testAcc2}%')
    print(f'Train dataset: {trainAcc2}%')
    print("The result for class 1 and class 3 prediction: ")
    print(f'Test dataset: {testAcc3}%')
    print(f'Train dataset: {trainAcc3}%')
    print("\n")

    # This time the dataset will not be randomised
    m1.train_perceptron( train1.feature_dataset, train1.label_dataset, 20, randomise=False)
    m2.train_perceptron(train2.feature_dataset, train2.label_dataset, 20, randomise=False)
    m3.train_perceptron(train3.feature_dataset, train3.label_dataset, 20, randomise=False)

    trainAcc1 = classify.compute_prediction_accuracy(train1.feature_dataset, train1.label_dataset, m1.get_bias(), m1.get_weight())
    trainAcc2 = classify.compute_prediction_accuracy(train2.feature_dataset, train2.label_dataset, m2.get_bias(), m2.get_weight())
    trainAcc3 = classify.compute_prediction_accuracy(train3.feature_dataset, train3.label_dataset, m3.get_bias(), m3.get_weight())
    
    testAcc1 = classify.compute_prediction_accuracy(test1.feature_dataset, test1.label_dataset, m1.get_bias(), m1.get_weight())
    testAcc2 = classify.compute_prediction_accuracy(test2.feature_dataset, test2.label_dataset, m2.get_bias(), m2.get_weight())
    testAcc3 = classify.compute_prediction_accuracy(test3.feature_dataset, test3.label_dataset, m3.get_bias(), m3.get_weight())
    
    print("Question 3 - Report the train and test classification accuracies:")
    print("The training dataset will not be randomised during each epoch.")
    print(Constant.BREAKPOINT)
    print("The result for class 1 and class 2 prediction: ")
    print(f'Test dataset: {testAcc1}%')
    print(f'Train dataset: {trainAcc1}%')
    print("The result for class 2 and class 3 prediction: ")
    print(f'Test dataset: {testAcc2}%')
    print(f'Train dataset: {trainAcc2}%')
    print("The result for class 1 and class 3 prediction: ")
    print(f'Test dataset: {testAcc3}%')
    print(f'Train dataset: {trainAcc3}%')
    print("\n")


    # ================================================================================================================
    # ================================================================================================================
    # Question 4 - Report the classification accuracies for 1-vs-rest approach
    # ================================================================================================================
    # ================================================================================================================

    # Load all train dataset. Each will have one class as positive and the rest negative
    # The training dataset with one class positive will be used for one-vs-rest perceptron classification. There will be k dataset for k classes
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(1) # class_1-vs-rest
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(2) # class_2-vs-rest
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(3) # class_3-vs-rest

    # Load the test dataset that will be used for classification accuracy
    test_data = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)

    # Load the train dataset that will be used for classification accuracy
    train_data = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False)

    # Initialise the perceptron algorithm to train our model
    model1 = Perceptron(0, 0)
    model2 = Perceptron(0, 0)
    model3 = Perceptron(0, 0)

    # Training the model
    model1.train_perceptron(train1.feature_dataset, train1.label_dataset, 20)   # class_1-vs-rest
    model2.train_perceptron(train2.feature_dataset, train2.label_dataset, 20)   # class_2-vs-rest
    model3.train_perceptron(train3.feature_dataset, train3.label_dataset, 20)   # class_3-vs-rest

    # Initialise the Perceptron one-vs-rest classification
    predict = PerceptronMultiClassification()

    test_res = predict.compute_multiclass_prediction_accuracy(test_data.feature_dataset, test_data.label_dataset, model1, model2, model3)
    train_res = predict.compute_multiclass_prediction_accuracy(train_data.feature_dataset, train_data.label_dataset, model1, model2, model3)
    print("Question 4 - Report the train and test classification accuracies for 1-vs-rest:")
    print("The training dataset were randomised during each epoch.")
    print(Constant.BREAKPOINT)
    print("This is the accuracy for test dataset using 1-vs-rest classification: ")
    print(f'{test_res}%')
    print("This is the accuracy for train dataset using 1-vs-rest classification: ")
    print(f'{train_res}%')
    print("\n")
    
    # We will not randomised the dataset
    model1.train_perceptron(train1.feature_dataset, train1.label_dataset, 20, False)   # class_1-vs-rest
    model2.train_perceptron(train2.feature_dataset, train2.label_dataset, 20, False)   # class_2-vs-rest
    model3.train_perceptron(train3.feature_dataset, train3.label_dataset, 20, False)   # class_3-vs-rest

    test_res = predict.compute_multiclass_prediction_accuracy(test_data.feature_dataset, test_data.label_dataset, model1, model2, model3)
    train_res = predict.compute_multiclass_prediction_accuracy(train_data.feature_dataset, train_data.label_dataset, model1, model2, model3)
    print("Question 4 - Report the train and test classification accuracies for 1-vs-rest:")
    print("The training dataset will not be randomised during each epoch.")
    print(Constant.BREAKPOINT)
    print("This is the accuracy for test dataset using 1-vs-rest classification: ")
    print(f'{test_res}%')
    print("This is the accuracy for train dataset using 1-vs-rest classification: ")
    print(f'{train_res}%')
    print("\n")

    # ================================================================================================================
    # ================================================================================================================
    # Question 5 - Report the classification accuracies for the l2 regularisation Perceptron
    # ================================================================================================================
    # ================================================================================================================

    # Loads the training dataset for 1-vs-rest. Therefore each will have a specified class as positive and the rest negative
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(1) # class_1-vs-rest
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(2) # class_2-vs-rest
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=True).make_one_class_positive(3) # class_3-vs-rest

    # Load the test and train dataset for classification
    test_dataset = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)
    train_dataset = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False)

    # Regularisation coefficient values we can use during the test
    REG_COEFFICIENT = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Initialise the multi-classification object
    multi_classification = PerceptronMultiClassification()

    # Training the three models with L2 coefficient 
    # =======================================================================

    print("Question 5 - Report the result of train and test data with l2 regularisation applied to Perceptron:")
    print(Constant.BREAKPOINT)

    # Iterate through each coefficient and try out different model with different l2 coefficient and see the result (i.e. accuracy)
    for coefficient in REG_COEFFICIENT:
        # Initialise the models
        model1 = PerceptronRegularisation(0, 0) 
        model2 = PerceptronRegularisation(0, 0)
        model3 = PerceptronRegularisation(0, 0)

        # Train the model, each will be trained for different 1-vs-rest dataset and with an l2 coefficient
        model1.train_perceptron_l2(train1.feature_dataset, train1.label_dataset, coefficient, 20)
        model2.train_perceptron_l2(train2.feature_dataset, train2.label_dataset, coefficient, 20)
        model3.train_perceptron_l2(train3.feature_dataset, train3.label_dataset, coefficient, 20)

        # Compute the accuracy of the prediction
        test_dataset_accuracy = multi_classification.compute_multiclass_prediction_accuracy(test_dataset.feature_dataset, test_dataset.label_dataset, model1, model2, model3)
        train_dataset_accuracy = multi_classification.compute_multiclass_prediction_accuracy(train_dataset.feature_dataset, train_dataset.label_dataset, model1, model2, model3)

        print(f'The result with regularisation coefficient of {coefficient}')
        print(f'Test dataset accuracy:  {test_dataset_accuracy}%')
        print(f'Train dataset accuracy: {train_dataset_accuracy}%')
        print("-----")
