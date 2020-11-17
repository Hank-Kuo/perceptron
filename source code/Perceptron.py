import numpy as np




class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, nueron_size,component_size , transfer_target):
        # Type: (int, ndarray) -> None
        """Initial of Perceptron Network
        Args :
            nueron_size : input the nueron size which you need
            transfer_traget : the Perceptron Classes match to the specfic array which I defined
                the array refer to the **group variable**  and **group_1 variable**

        Attributes :
            W : ndarray, A Nx3 Matrix of Perceptron Network's Weight.
            B : A Nx3 Matrix of Perceptron Network's Bias.
            epochs : The epochs of Perceptron Network.
            iterator : An integer count of every epochs.
            lr : The learning rate of Perceptron Network.
            e : Counting error number which is non-zero matrix in every epochs.
            transfer_traget : the Perceptron Classes match to the specfic array which I defined
                the array refer to the **group variable**  and **group_1 variable**
            labels : the Class of Perceptron Network.

        Type Attributes :
            (ndarray, ndarray, int, int, int, ndarray, list)
        """
        self.W = np.zeros((nueron_size,component_size))
        self.B = np.zeros((nueron_size, 1))
        self.epochs = 0
        self.iterator = 0
        self.lr = 1
        self.e = np.zeros((nueron_size, 1))+1
        self.transfer_target = transfer_target
        self.labels = ['W', 'P', 'O', 'B']

    def activation_fn(self, x):
        # Type : (ndarray) -> ndarray
        """Activation function by Hard Limit
        Processed the Matrix by activation function using Hard Limit.
        The Hard Limit means the Matrix values more or equal than 0 is 1, and the
        Matrix vaules less than 0 is 0.

        Args:
            x : A numpy of Matrix which need to determine the value of Matrix is greater
                than 0 or not.
        Returns:
            x : A numpy of Matrix which are already proecessed.
        """

        x[x >= 0] = 1
        x[x < 0] = 0
        return x

    def output(self, data):
        # Type : (ndarray) -> ndarray
        """Output for Perceptron
        Processing the data from training data or testing data.
        Data will be dot by W and plus B, then generate the net_input, the
        net_input through activation funciton will finally generate output a.

        Args:
            data : A Numpy of Matrix need to dot by Weight and plus Bias, and the
                net_input through activation funciton

        Return :
            a : A final output of Numpy of Matrix which is already proecessed.
        """
        net_input = self.W.dot(data) + self.B
        a = self.activation_fn(net_input)
        return a

    def transfer_list(self, target):
        # Type : (String) -> ndarray
        """Transfer Traget to array
        Let target class transfer to Matrix which is I defined
        Args :
            target : All Class like ("W", "P", "B", "O") corresponed to specific Matrix

        Attribues:
            index : Find the Class's index in the labels list

        Return :
            a : A final output of Numpy of Matrix which is already proecessed
        """
        index = self.labels.index(target)
        a = np.array([self.transfer_target[index]]).T
        return a

    def train(self, data, component):
        # Type : (ndarray) -> None
        """Training Perceptron network
        Load Testing data to train Perceptron Network, then if epochs >= 1000 or
        error eual to lengh of data will stop training. The whole process which get
        2 componet ("shape", "texture") to train, and using self.e to adjust Weight
        and Bias in Perceptron.

        Args :
            data :

        Attributes :
            error : An Integer count which judge self.e equal to lengh of data
        """

        error = 0
        while(error != len(data) and self.epochs < 100):
            error = 0
            for i in range(len(data)):
                origin = data[component][i:i+1].values.T
                target = data.Class.values[i]
                a = self.output(origin)
                self.e = self.transfer_list(target) - a
                self.W = self.W + self.lr * self.e * origin.T
                self.B = self.B + self.lr * self.e
                self.iterator += 1
                if(not np.any(self.e)):
                    error += 1
            self.epochs += 1

    def predict(self, data, component):
        # Type : (ndarray) -> None
        """Predict Perceptron network
        Loaing all data which need to predict, get two component ("shape","texture")and
        through output function, then finding which matches the transfer_target, then show
        the corresponding labels

        Args :
            data : The whole of Matrix which need to predict using Perceptron network

        Attributes :
            result : All output which is predicted by Perceptron Network
            origin : Original Data which be pre-processing from data
            a : The origin dot Weight and plus Bais, and through activate function , then
                generate finally output
        """
        result = "Testing: "
        for i in range(len(data)):
            origin = data[component][i:i+1].values.T
            a = self.output(origin).T
            flag = True
            for j in range(len(self.transfer_target)):
                if np.alltrue(self.transfer_target[j] == a[0]):
                    result += str(i+1)+str(self.labels[j])+" "
                    flag = False
            if flag :
                result += str(i+1)+"No"+" "

        print(result)

