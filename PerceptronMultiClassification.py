import numpy as np
from DatasetHandler import DatasetHandler
from Perceptron import Perceptron
from PerceptronClassification import PerceptronClassification
from Constant import Constant, view_dataset 


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
    # Question 4
    # Load all train dataset. Each will have one class as positive and the rest negative
    # The training dataset with one class positive will be used for one-vs-rest perceptron classification. There will be k dataset for k classes
    train1 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(1) # class_1-vs-rest
    train2 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(2) # class_2-vs-rest
    train3 = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False).make_one_class_positive(3) # class_3-vs-rest

    # Load the test dataset that will be used for classification accuracy
    test_data = DatasetHandler().extract_all_classes_from_dataset('test.data', randomise=False)

    # Load the train dataset that will be used for classification accuracy
    train_data = DatasetHandler().extract_all_classes_from_dataset('train.data', randomise=False)

    # view_dataset(train_class3_x, train_class3_y)

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
    print(test_res)
    print(train_res)
    