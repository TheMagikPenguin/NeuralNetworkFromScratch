import math
weights = [-2,-2]
bias = 3
inputs = input("Enter input: ")
inputs = [int(c) for c in inputs]
def dotProduct(l1, l2):
    product = 0
    for i in range(len(l1)):
        product += l1[i]*l2[i]
    return product
def calculateZ(bias):
    value = dotProduct(inputs, weights)
    value += bias
    return value
def sigmoidActivationFunction(z):
    return float(1/(1+math.pow(math.e, z*(-1))))
print(sigmoidActivationFunction(calculateZ(bias)))