"""Perceptron model."""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, dimension : int, batch_size : int, lambd : float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        np.random.seed(1)
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.d = dimension + 1
        self.batch_size = batch_size
        self.w = np.random.uniform(-1, 1, (n_class,self.d)) * 0.01 
        self.lambd = lambd

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        #shuffle data
        #for each epoch
        #decay learning rate
        #split into batches of size 100
        #for each batch:
        #initialize gradient vector to all 0's
        #for each example:
        #update gradient vector
        #then do update step
        
        #normalize
        X_train_augmented = X_train - np.mean(X_train, axis = 0)
        X_train_augmented = X_train_augmented / np.std(X_train_augmented, axis=0)

        X_train_augmented = np.hstack((X_train_augmented, np.ones((X_train_augmented.shape[0], 1))))
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
            W_grad = np.zeros((self.n_class,self.d))
            size = batch_X.shape[0]
            predictions = np.dot(self.w, batch_X.T) #gives a n_class by size vector
            for i in range(size): #go through each example
              label = batch_Y[i] #true class
              misclassified = 0 #number of misclassifications
              y_score = predictions[label][i]
              for c in range(self.n_class): #go through all the classes
                if c != label:
                  c_pred = predictions[c][i]
                  if (c_pred > y_score): #will lead to misclassification
                    misclassified += 1
                    W_grad[c] += batch_X[i]
              W_grad[label] -= misclassified * batch_X[i]  

            self.w = self.w - self.lr * (self.lambd / size * self.w + W_grad) #update step
          predictions_train = self.predict(X_train)
          accuracy = self.get_acc(predictions_train, y_train)
          print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.4f}")

          loss = self.computeLoss(X_train, y_train)
          print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")
          accuracies.append(accuracy)
          losses.append(loss)

        
        self.plot_accuracy(accuracies)
        self.plot_accuracy(losses)
        

    def plot_accuracy(self, accuracies: list):
        plt.plot(range(1, self.epochs + 1), accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Y')
        plt.title('Title')
        plt.grid(True)
        plt.show()

    def computeLoss(self, X_train, y_train):
      X_train_augmented = X_train - np.mean(X_train, axis = 0)
      X_train_augmented = X_train_augmented / np.std(X_train_augmented, axis=0)
      X_train_augmented = np.hstack((X_train_augmented, np.ones((X_train_augmented.shape[0], 1))))
      size = X_train_augmented.shape[0]
      loss = 0
      loss += self.lambd / 2 * np.sum(np.linalg.norm(self.w, axis=1) ** 2)
      predictions = np.dot(self.w, X_train_augmented.T)
      for i in range(size): #go through each example
              label = y_train[i] #true class
              y_score = predictions[label][i]
              for c in range(self.n_class): #go through all the classes
                  c_pred = predictions[c][i]
                  if (c_pred > y_score): #will lead to misclassification
                    loss += c_pred - y_score
      return loss

                

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test_augmented = X_test - np.mean(X_test, axis = 0)
        X_test_augmented = X_test_augmented / np.std(X_test_augmented, axis=0)

        X_test_augmented = np.hstack((X_test_augmented, np.ones((X_test_augmented.shape[0], 1))))
        
        return np.argmax((np.dot(self.w, X_test_augmented.T)),axis = 0)

    def get_acc(self, pred, y_test):
      return np.sum(y_test == pred) / len(y_test) * 100
