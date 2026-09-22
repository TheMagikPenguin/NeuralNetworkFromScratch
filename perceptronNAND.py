weights = [-2, -2]
bias = 3
inputs = input("Enter input: ")
inputs = [int(c) for c in inputs]
def dotProduct(l1, l2):
    product = 0
    for i in range(len(l1)):
        product += l1[i]*l2[i]
    return product
def perceptron(bias):
    value = dotProduct(inputs, weights)
    value += bias
    if value > 0:
        return 1
    if value <= 0:
        return 0
print(perceptron(bias))