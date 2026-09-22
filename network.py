import random, math, struct
import tkinter as tkin

'''Just plugs z (an input vector) into the sigmoid activation function'''
def sigmoidFunction(z):
    outputVector = []
    for i in z:
        outputVector.append((1.0)/(1.0+math.exp(-i)))
    return outputVector
def sigmoidFunctionDerivative(z):
    outputVector = []
    for i in z:
        sigma = (1.0)/(1.0+math.exp(-i))
        outputVector.append(sigma * (1 - sigma))
    return outputVector

class Network:
    '''dim is a list of what dimension each neural layer lives in basically dim = [784, 15, 10] indicates 784 neurons
       in the input layer, 15 in the hidden layer, and 10 in the output layer.'''
    def __init__(self, dim):
        self.numLayers = len(dim)
        self.layerSizes = dim

        '''bias vector (this gives the starting point for the gradient descent)'''
        self.biases = []

        '''loop through each network layer excluding the input layer.'''
        for L in dim[1:]:
            currentLayerBiases = []
            '''for every neuron in that layer create a bias on it which is just a random number from a standard
               distribution curve. It also uses the same probabilities for the values.'''
            for b in range(L):
                currentLayerBiases.append(random.gauss(0,1))
            self.biases.append(currentLayerBiases)
        '''Now the structure of the bias vector is: [b1, b2, b3....] where {b1, b2, b3} are all column vectors each one
           representative of the biases in a neural layer'''
        
        '''weights vectors. Its a 2d array each column is the weights for the neuron in that layer structured like this
            [
                [n11, n12, n13],
                [n21, n22, n23],
                [n31, n32, n33]
            ]
            In this case the embedded lists represent a list of each neuron in a layer and the elements
            n11, n12, n12, n21.... are all column vectors that represent the weights of that neuron connecting to
            the next layer'''
        self.weights = []
        for i in range(len(dim) - 1):
            # x: the input layer to a connection
            # y: the output layer to a connection
            x = dim[i]
            y = dim[i+1]
            currentLayerWeights = []
            for j in range(y):
                r = []
                for k in range(x):
                    r.append(random.gauss(0,1))
                currentLayerWeights.append(r)
            self.weights.append(currentLayerWeights)
    
    '''in a nutshell this function is just doing a lot of sigmoid function and passing the resulting vector forward
       to the end of the network'''
    def feedForward(self, activations):
        for biases, weights in zip(self.biases, self.weights):
            z = []
            for neuronBias, neuronWeights in zip(biases, weights):
                weightedSum = 0
                for i in range(len(activations)):
                    weightedSum += activations[i]*neuronWeights[i]
                weightedSum += neuronBias
                z.append(weightedSum)
            activations = sigmoidFunction(z)
        return activations
    
    def stochasticGradientDescent(self, trainData, epochs, batchSize, learningRate, testData = None):
        numOfTrain = len(trainData)
        if testData:
            numOfTest = len(testData)
        for j in range(epochs):
            random.shuffle(trainData)
            miniBatches = []
            for k in range(0, numOfTrain, batchSize):
                miniBatches.append(trainData[k : k + batchSize])
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, learningRate)
            if testData:
                r = self.evaluateCorrectness(testData)
                print(f'Epoch {j} accuracy rate: {r} / {numOfTest}')
            else:
                print(f'Epoch {j} finished')
    
    def backprop(self, x, y):
        delB = []
        for b in self.biases:
            #This is just a list of zeros for the current layer
            layerZeros = []
            for i in b:
                layerZeros.append(0.0)
            delB.append(layerZeros)
        delW = []

        for w in self.weights:
            #This is just a list of zeros for the current layer
            layerZeros = []
            for neuronWeights in w:
                neuronZeros = []
                for i in neuronWeights:
                    neuronZeros.append(0.0)
                layerZeros.append(neuronZeros)
            delW.append(layerZeros)
        #feedforward
        activation = x
        activations = [x]
        zVectors = []
        for layer in range(len(self.biases)):
            biasLayer = self.biases[layer]
            weightLayer = self.weights[layer]
            z = []
            for neuron in range(len(weightLayer)):
                weightSum = 0.0
                for prevNeuron in range(len(weightLayer[neuron])):
                    weightSum += weightLayer[neuron][prevNeuron] * activation[prevNeuron]
                weightSum += biasLayer[neuron]
                z.append(weightSum)
            zVectors.append(z)
            activation = sigmoidFunction(z)
            activations.append(activation)
        #Find the delta(change in) my output layer vector
        costDerivative = self.costDerivative(activations[-1], y)
        sigmoidDerivativeLastLayer = sigmoidFunctionDerivative(zVectors[-1])
        delta = []
        for i in range(len(costDerivative)):
            delta.append(costDerivative[i] * sigmoidDerivativeLastLayer[i])
        #Fill out the delB and delW for the last layer the delB will just use delta and delW just multiply delta
        #and activations[finalHiddenLayer] (nOTE: this is an outer product NOT hadamard product (I messed that up b4))
        delB[-1] = delta
        for neuron in range(len(delW[-1])):
            for prevNeuron in range(len(delW[-1][neuron])):
                delW[-1][neuron][prevNeuron] = delta[neuron] * activations[-2][prevNeuron]
        #Move backward through hidden layers applying blame to which neuron ruined the signal the most which will be our DelB and DelW we return
        for layer in range(2, self.numLayers):
            z = zVectors[-layer]
            sigmoidDerivativeCurrentLayer = sigmoidFunctionDerivative(z)
            nextWeights = self.weights[-layer + 1]
            newDelta = []

            for neuron in range(len(z)):
                weightedDeltaSum = 0.0
                for nextNeuron in range(len(delta)):
                    weightedDeltaSum += nextWeights[nextNeuron][neuron] * delta[nextNeuron]
                newDelta.append(weightedDeltaSum * sigmoidDerivativeCurrentLayer[neuron])
            delta = newDelta
            delB[-layer] = delta
            for neuron in range(len(delW[-layer])):
                for previousNeuron in range(len(delW[-layer][neuron])):
                    delW[-layer][neuron][previousNeuron] = delta[neuron] * activations[-layer - 1][previousNeuron]
        return delB, delW
    def updateMiniBatch(self, miniBatch, learningRate):
        delB = []
        for b in self.biases:
            #This is just a list of zeros for the current layer
            layerZeros = []
            for i in b:
                layerZeros.append(0.0)
            delB.append(layerZeros)
        delW = []
        for w in self.weights:
            #This is just a list of zeros for the current layer
            layerZeros = []
            for neuronWeights in w:
                neuronZeros = []
                for i in neuronWeights:
                    neuronZeros.append(0.0)
                layerZeros.append(neuronZeros)
            delW.append(layerZeros)
        for x, y in miniBatch:
            deltaDelB, deltaDelW = self.backprop(x,y)
            #Constructing bias gradients(the dels for b vector)
            for layer in range(len(delB)):
                for neuron in range(len(delB[layer])):
                    delB[layer][neuron] += deltaDelB[layer][neuron]
            #Constructing weight gradients (the dels for w vector)
            for layer in range(len(delW)):
                for neuron in range(len(delW[layer])):
                    for weight in range(len(delW[layer][neuron])):
                        delW[layer][neuron][weight] += deltaDelW[layer][neuron][weight]
        batchSize = len(miniBatch)
        #Use the delB and delW we just made to update the actual biases and weights
        for layer in range(len(self.biases)):
            for neuron in range(len(self.biases[layer])):
                self.biases[layer][neuron] -= (learningRate / batchSize) * delB[layer][neuron]
        for layer in range(len(self.weights)):
            for neuron in range(len(self.weights[layer])):
                for weight in range(len(self.weights[layer][neuron])):
                    self.weights[layer][neuron][weight] -= (learningRate / batchSize) * delW[layer][neuron][weight]

    def costDerivative(self, outputActivations, yVector):
        derivativeOutput = []
        for i in range(len(outputActivations)):
            derivativeOutput.append(outputActivations[i] - yVector[i])
        return derivativeOutput
    
    def evaluateCorrectness(self, testData):
        correct = 0
        for x, y in testData:
            output = self.feedForward(x)
            maxIndex = 0
            #This just traverses through the output layer and finds the index of the highest activation
            for i in range(len(output)):
                if output[i] > output[maxIndex]:
                    maxIndex = i
            #If what the network guessed matches the actual digit then it gets +1 correct
            if maxIndex == y:
                correct+=1
        return correct
    
    def guess(self, x):
        answers = self.feedForward(x)
        maxIndex = 0
        for i in range(len(answers)):
            if answers[i]>answers[maxIndex]:
                maxIndex = i
        return maxIndex, answers

def trueAnswerVector(label):
    #makes a 10 entry list of 0.0s
    output = [0.0]*10
    #applies a 1.0 to the index that corresponds to the actual digit
    output[label] = 1.0
    return output

def translateImages(filename):
    images = []
    "Opening with rb just means 'reading binary' which means instead of reading like a text file its just reading in binary numbers"
    with open(filename, "rb") as file:
        """MNIST IDX IMAGE 16 BYTE STRUCTURE
        Bytes 0-3: Magic Number that signifies if it is label or image
        Bytes 4-7: Identifies the how many items are in the file (# of images or labels)
        Bytes 8-11: These bytes are always 00 00 00 1C which equal 28 and it signifies the number of pixel rows
        Bytes 12-15: These bytes are always 00 00 00 1C which equal 28 and it signifies the number of pixel columns"""
        #Unpacking with >IIII: the > operator signifies the data is in big endian byte order cause it is just check the documentation lol
        # And the 4 I's signify there will be a tuple of 4 integers each requiring 4 bytes which matchs the outlined MNIST IDX Image 16 byte structure above.
        #file.read(16) signifies just read 16 bytes since we are in rb mode
        magicNum, numberOfItems, r, c = struct.unpack(">IIII", file.read(16))
        for imageIndex in range(numberOfItems):
            image = []
            """This traversal seems weird but struct.unpack was already moving through the file in rc form so I just figured
            a traversale where I just use range r*c would make more sense than nested for loops"""
            for pixelIndex in range(r*c):
                #This line unpacks a single byte in big endian using ">B" and file.read(1) only selects 1 byte
                #This byte ranges from 0 to 255 in denary which can be a grayscale
                #I added the [0] because unpack returns a tuple and I only require the first item
                pixel = struct.unpack(">B", file.read(1))[0]
                #I append the pixel to the array with its darkness being represented as 0.0-1.0 so it can be passable data into my network.
                image.append(pixel/255.0)
            images.append(image)
        return images

def translateLabels(filename):
    labels = []
    
    with open(filename, "rb") as file:
    #If we're unpacking a label there will only be 8 bytes because the other 8 needed for the pixels are empty
        magicNum, numberOfLabels = struct.unpack(">II", file.read(8))
        for labelIndex in range(numberOfLabels):
            label = struct.unpack(">B", file.read(1))[0]
            labels.append(label)
    return labels

def loadMNISTDatabase():
    testingImages = translateImages("t10k-images.idx3-ubyte")
    testingLabels = translateLabels("t10k-labels.idx1-ubyte")
    trainingImages = translateImages("train-images.idx3-ubyte")
    trainingLabels = translateLabels("train-labels.idx1-ubyte")

    trainingData = []
    for i in range(len(trainingImages)):
        x, y = trainingImages[i], trueAnswerVector(trainingLabels[i])
        trainingData.append((x, y))
    
    testingData = []
    for i in range(len(testingImages)):
        x, y = testingImages[i], testingLabels[i]
        testingData.append((x, y))
    return trainingData, testingData

def printImageASCII(img):
    for r in range(28):
        line = ""
        for c in range(28):
            pixel = img[r*28 + c]
            #if fully black the pixel value will just be 0 and if anything else itll mult the float by 255 and then intround
            pixelValue = int(pixel* 255)
            if pixelValue < 10:
                line += "  " + str(pixelValue) + " "
            elif pixelValue < 100:
                line += " " + str(pixelValue) + " "
            else:
                line += str(pixelValue) + " "
        print(line)

def makeTkinterWindow():
    window = tkin.Tk()
    label = tkin.Label(window)
    label.pack()
    return window, label

def printImageTkinter(window, label, img):
    scale = 15
    image = tkin.PhotoImage(width=28, height=28)
    for r in range(28):
        for c in range(28):
            pixel = img[r*28 + c]
            pixelValue = int(pixel*255)
            hexValue = hex(pixelValue)[2:]
            if len(hexValue) == 1:
                hexValue = "0"+hexValue
            color = "#" + hexValue + hexValue + hexValue
            image.put(color, (c, r))
    enlargedImage = image.zoom(scale, scale)
    label.configure(image=enlargedImage)
    label.image = enlargedImage
    window.update()

#---------------------------------------------------
#THE GRAND RUN LOOP FOLLOWING MY CODE METHOD VOMIT
#---------------------------------------------------

if __name__ == "__main__":
    #crashes if not utf-8 because the title has special characters windows doesn't recognize by default
    with open("title.txt", "r", encoding = "utf-8") as file:
        print(f'{file.read()}\n')
    print('Loading the training data so the model can actually train and guess gimme a second (～￣▽￣)～\n')
    trainingData, testingData = loadMNISTDatabase()
    """784 input neurons (data from 784 pixels to input),
       20 hidden layer neurons (its lower that recommended amounts because efficiency is needed with my inefficientmath),
       and 10 output layer neurons each representing 10 digits"""
    network = Network([784, 20, 10])
    print("Data loaded successfully now I will begin training ᕦ(ò_óˇ)\n-------------------------------------------------------------------------------------------------------------------------\n")
    network.stochasticGradientDescent(trainingData[:4000], epochs=10, batchSize=10, learningRate=3, testData=testingData[:200])
    imageWindow, imageLabel = makeTkinterWindow()
    while 1==1:
        guessingImageIndex = random.randint(0, len(testingData) - 1)
        guessingImage = testingData[guessingImageIndex][0]
        actualAnswer = testingData[guessingImageIndex][1]
        guess, answer = network.guess(guessingImage)
        print("\nHeres the image I'll guess ( •̀ ω •́ )✧:")
        printImageASCII(guessingImage)
        printImageTkinter(imageWindow, imageLabel, guessingImage)
        print(f'\nI think that this a picture of a: {guess}')
        print(f'\nThe actual picture was: {actualAnswer}')
        if guess == actualAnswer:
            print("I GOT ITTTTTTTTT *^____^*")
        else:
            print("Oops i got it wrong.... (┬┬﹏┬┬)")
        cont = input("\nPress ENTER for another guess, or type q to quit: ")
        if cont.lower() == "q":
            break
    imageWindow.destroy()