from random import random
from util import matmul, matshape, relu

class RegressorModel():
    """
    2 layer MLP for regression problem
    """
    def __init__(self, n: int, d: int, k: int, apply_bias=True):
        """
        Model initialization

        Parameters: 
        n: int - input feature size
        d: int - hidden layer size
        k: int - output layer size
        """
        self.bias = apply_bias
        self.W1 = [[random()]*d for _ in range(n)] 
        self.W2 = [[random()]*k for _ in range(d)]

        if self.bias:
            self.W1.append([0] * d)
            for row in self.W1:
                row.append(0)
            self.W2.append([0] * k)

    def forward(self, X):
        """
        Forward pass of a model, expects X as matrix of shape m x n

        ### Parameters:
        X - 2D Matrix of shape m x n

        ### Returns:
        Z - vector of length m, regressed values
        """
        if self.bias:
            for row in X:
                row.append(1)
        # X @ W1
        H1 = matmul(X, self.W1)

        # RelU(X @ W1)
        for i in range(len(H1)):
            relu(H1[i])
        
        # ReLU(X @ W1) @ W2
        Z = matmul(H1, self.W2)

        # ReLU(ReLU(X @ W1) @ W2)
        for i in range(len(H1)):
            relu(Z[i])
        
        # sum by row
        output = []
        for row in Z:
            output.append(sum(row))
        
        return output

    def __str__(self):
        return  f"RegressorModel\n----------------\nW1{matshape(self.W1)}\nW2{matshape(self.W2)}\n"