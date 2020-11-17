from Perceptron import Perceptron
import numpy as np
import pandas as pd

# Dataset 2 group
group = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
group_1 = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])

dataset2_group_2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

dataset2_group_4 = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])

# Dataset 1 and Datset 1 group
training_dataset1 = np.array([ [1,1,1], [1,2,1], [2,-1,2], [2,0,2],
                            [-1, 2, 3], [-2, 1, 3], [-1, -1, 4], [-2, -2, 4]])

testing_dataset1 = np.array([ [5,2], [0,-2], [-1,1], [-3,-4] ])

dataset1_group_2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

dataset1_group_4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0 ], [0, 0, 1, 0 ], [0, 0, 0, 1 ]])

if __name__ == '__main__':
    # Loading Training Data and Testing Data
    training_data = pd.read_csv('改_training_data.txt', sep=' ', names=[
                                "shape", "texture", "weight", "Class"])
    testing_data = pd.read_csv('改_testing_data.txt', sep=' ', names=[
                               "shape", "texture", "weight"])
    training_data_1 = pd.DataFrame(training_dataset1, columns=["P1", "P2", "Class"])
    testing_data_1 = pd.DataFrame(testing_dataset1,columns=["P1","P2"])


    # component
    dataset1_component = ["P1","P2"]
    dataset2_two_component = ["shape", "texture"]
    dataset2_three_component = ["shape", "texture","weight"]



    # Train four neuron Perceptron Model with two component -- Dataset 1
    perceptron = Perceptron(nueron_size=4,component_size=2,transfer_target=dataset1_group_4)
    perceptron.labels = [1, 2, 3, 4]
    print("(b)Four-neuron perceptron: (1)Dataset 1")
    print("------------------------------------------------------")
    print("Initial")
    print("Weight:")
    print(perceptron.W)
    print("Biases:")
    print(perceptron.B)
    perceptron.train(training_data_1, dataset1_component)
    print("After Traning")
    print("Weight:")
    print(perceptron.W)
    print("Biases:")
    print(perceptron.B)
    print("Epoch: "+ str(perceptron.epochs))
    perceptron.predict(testing_data_1 , dataset1_component )

    # Train four neuron Perceptron Model with two component -- Dataset 2
    perceptron1 = Perceptron(nueron_size=4,component_size = 2 , transfer_target=dataset2_group_4)
    print("(b)Four-neuron perceptron: (2)Dataset 2 – Use the first two components.")
    print("------------------------------------------------------")
    print("Initial")
    print("Weight:")
    print(perceptron1.W)
    print("Biases:")
    print(perceptron1.B)
    perceptron1.train(training_data, dataset2_two_component)
    print("After Traning")
    print("Weight:")
    print(perceptron1.W)
    print("Biases:")
    print(perceptron1.B)
    print("Epoch: "+ str(perceptron1.epochs))
    perceptron1.predict(testing_data, dataset2_two_component )

    # Train four neuron Perceptron Model with three component -- Dataset 2
    perceptron2 = Perceptron(nueron_size=4,component_size=3 , transfer_target=dataset2_group_4)
    print("(b)Four-neuron perceptron: (3)Dataset 2 – Use the three components.")
    print("------------------------------------------------------")
    print("Initial")
    print("Weight:")
    print(perceptron2.W)
    print("Biases:")
    print(perceptron2.B)
    perceptron2.train(training_data, dataset2_three_component)
    print("After Traning")
    print("Weight:")
    print(perceptron2.W)
    print("Biases:")
    print(perceptron2.B)
    print("Epoch: "+ str(perceptron2.epochs))
    perceptron2.predict(testing_data, dataset2_three_component )
