"""Logistic regression model."""

import numpy as np
import matplotlib.pyplot as plt


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, dimension : int, batch_size : int, lambd : float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        
        self.threshold = threshold
        np.random.seed(1)
        self.lr = lr
        self.epochs = epochs
        self.d = dimension + 1
        self.batch_size = batch_size
        self.w = np.random.uniform(-1, 1, (1,self.d)) * 0.01 
        self.lambd = lambd

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.

        z = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        return z
        

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        #fix y labels to -1 and 1
        y_train_original = y_train
        y_train = np.where(y_train == 0, -1, 1)

        X_train_augmented = X_train - np.mean(X_train, axis = 0)
        X_train_augmented = X_train_augmented / np.std(X_train_augmented, axis=0)

        X_train_augmented = np.hstack((X_train_augmented, np.ones((X_train_augmented.shape[0], 1)))) #add bias
        N = X_train_augmented.shape[0]
        D = X_train_augmented.shape[1]
        initial_lr = self.lr
        accuracies = []
        losses = []

        indices = np.random.permutation(N)
        shuffled_X_train = X_train_augmented[indices]
        shuffled_Y_train = y_train[indices]
        for epoch in range (self.epochs):
          self.lr *= 0.4
          for b in range(0, N, self.batch_size):
            batch_X = shuffled_X_train[b:b+self.batch_size]
            batch_Y = shuffled_Y_train[b:b+self.batch_size]
            W_grad = np.zeros((1,self.d))
            size = batch_X.shape[0]
            predictions = np.dot(self.w, batch_X.T) #gives a 1 by size vector
            #calculate sigmoid
            predictions = predictions * -1 * batch_Y
            predictions = self.sigmoid(predictions)

            for i in range(size): #go through each example
              W_grad[0] += -1 * predictions[0][i] * batch_Y[i] * batch_X[i]
            self.w = self.w - self.lr * (self.lambd / size * self.w + W_grad) #update step
          predictions_train = self.predict(X_train)
          accuracy = self.get_acc(predictions_train, y_train_original)
          print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.4f}")

         
          accuracies.append(accuracy)

        
        self.plot_accuracy(accuracies)
        self.plot_accuracy(losses)
        

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test_augmented = X_test - np.mean(X_test, axis = 0)
        X_test_augmented = X_test_augmented / np.std(X_test_augmented, axis=0)

        X_test_augmented = np.hstack((X_test_augmented, np.ones((X_test_augmented.shape[0], 1))))
        predictions = np.dot(self.w, X_test_augmented.T)
        predictions = self.sigmoid(predictions)
        return np.where(predictions >= 0.5, 1, 0)

    def plot_accuracy(self, accuracies: list):
        # plt.plot(range(1, self.epochs + 1), accuracies)
        # plt.xlabel('Epochs')
        # plt.ylabel('Y')
        # plt.title('Title')
        # plt.grid(True)
        # plt.show()
        pass

    
    def get_acc(self, pred, y_test):
      return np.sum(y_test == pred) / len(y_test) * 100

    
