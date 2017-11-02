import numpy as np
import PIL
import PIL.Image
import cv2
import multiLayerNeuralNetwork

PREFIX = "/Users/rasrivastava/neural_net/code/files/train/train/"

PERCENT = 1
IMAGE_SIZE = (100,100)
def prepareInputForCat(catFileName):
    X = np.zeros((1,1))
    assigned = False
    count = 0
    with open(catFileName) as f:
        for line in f:
            count = count + 1
            val = np.random.randint(0,100)
            if (val < PERCENT):
                imageFile = PREFIX + line
                imageFile = imageFile.replace("\n","")
                arr = np.asarray(PIL.Image.open(imageFile))
                updatedArr = cv2.resize(arr, IMAGE_SIZE)
                #print("updated are " + str(updatedArr.shape))
                x = updatedArr.flatten()
                x = x/255.0
                y = np.reshape(x,(x.shape[0],1))
                #print("y shape " + str(y.shape))
                x = y

                if (assigned == False):
                    X = x
                    assigned = True
                else:
                    X = np.column_stack((X,x))
                    #print("Xs state " + str(X.shape))
            else:
                #print("Skip sample " + str(count))
                count = 0

    return X


def prepareInputForDog(dogFileName):
    Y = np.zeros((1,1))
    assigned = False
    with open(dogFileName) as f:
        for line in f:
            val = np.random.randint(0,100)
            if (val < PERCENT):
                imageFile = PREFIX + line
                imageFile = imageFile.replace("\n","")
                arr = np.asarray(PIL.Image.open(imageFile))
                updatedArr = cv2.resize(arr, IMAGE_SIZE)
                y = updatedArr.flatten()
                y = np.reshape(y,(y.shape[0],1))
                #y = (y - np.mean(y))/np.std(y)
                y = y/255.0
                if (assigned == False):
                    Y = y
                    assigned = True
                    #print("After assignment " + str(X.shape))
                else:
                    #print(Y.shape)
                    Y = np.column_stack((Y,y))

    return Y




def formInputOutput():
    X1 = prepareInputForCat("/Users/rasrivastava/neural_net/code/files/train/train/catFile.txt")
    X2 = prepareInputForDog("/Users/rasrivastava/neural_net/code/files/train/train/dogFile.txt")


    print("X1 shape " + str(X1.shape))
    print("X2 shape " + str(X2.shape))
    Y1 = np.ones((1,X1.shape[1]))
    Y2 = np.zeros((1,X2.shape[1]))
    print("Y1.shape " + str(Y1.shape))
    print("Y2 shape " + str(Y2.shape))

    X = np.concatenate((X1,X2),axis=1)

    Y = np.concatenate((Y1,Y2),axis=1)
    return ((X*1.0),Y*1.0)

def formInputOutputRandom():
    X1 = prepareInputForCat("/Users/rasrivastava/neural_net/code/files/train/train/catFile.txt")
    X2 = prepareInputForDog("/Users/rasrivastava/neural_net/code/files/train/train/dogFile.txt")

    #Y1 = np.ones((1,X1.shape[1]))
    #Y2 = np.zeros((1,X2.shape[1]))
    #
    # X = np.concatenate((X1,X2),axis=1)
    #
    # Y = np.concatenate((Y1,Y2),axis=1)
    x1_index = -1
    x2_index = -1
    X_updated = np.zeros((X1.shape[0],1))
    Y_updated = np.zeros((1,1))
    while (x1_index < X1.shape[1]-1 and x2_index < X2.shape[1]-1):
        r  = np.random.uniform(0,1)
        #print(str(x1_index) + " " + str(x2_index) + " " + str(X1.shape[1]) + " "  + str(X2.shape[1]))
        if (r > 0.5):
            x1_index = x1_index+1
            extracted_matrix = X1[:,x1_index]
            extracted_matrix_shape = extracted_matrix.shape
            #print(extracted_matrix_shape)
            extracted_matrix_shape = (extracted_matrix_shape[0],1)
            extracted_matrix = extracted_matrix.reshape(extracted_matrix_shape)
            X_updated = np.column_stack((X_updated,extracted_matrix))
            Y_updated = np.column_stack((Y_updated,np.ones((1,1))))
        else:
            x2_index = x2_index + 1
            extracted_matrix = X2[:,x2_index]
            extracted_matrix_shape = extracted_matrix.shape
            #print(extracted_matrix_shape)
            extracted_matrix_shape = (extracted_matrix_shape[0],1)
            extracted_matrix = extracted_matrix.reshape(extracted_matrix_shape)
            X_updated = np.column_stack((X_updated,extracted_matrix))
            Y_updated = np.column_stack((Y_updated,np.zeros((1,1))))

    if x1_index < X1.shape[1]-1:
        while (x1_index < X1.shape[1]-1):
            x1_index = x1_index+1
            extracted_matrix = X1[:,x1_index]
            extracted_matrix_shape = extracted_matrix.shape
            #print(extracted_matrix_shape)
            extracted_matrix_shape = (extracted_matrix_shape[0],1)
            extracted_matrix = extracted_matrix.reshape(extracted_matrix_shape)
            X_updated = np.column_stack((X_updated,extracted_matrix))
            Y_updated = np.column_stack((Y_updated,np.ones((1,1))))

    if x2_index < X2.shape[1]-1:
        while (x2_index < X2.shape[1]-1):
            x2_index = x2_index + 1
            extracted_matrix = X2[:,x2_index]
            extracted_matrix_shape = extracted_matrix.shape
            #print(extracted_matrix_shape)
            extracted_matrix_shape = (extracted_matrix_shape[0],1)
            extracted_matrix = extracted_matrix.reshape(extracted_matrix_shape)
            X_updated = np.column_stack((X_updated,extracted_matrix))
            Y_updated = np.column_stack((Y_updated,np.zeros((1,1))))



    return ((X_updated*1.0),Y_updated*1.0)


def predict(W,B,imageFile):
    arr = np.asarray(PIL.Image.open(imageFile))
    updatedArr = cv2.resize(arr, IMAGE_SIZE)
    x = updatedArr.flatten()
    x = x/255.0
    y = np.reshape(x,(x.shape[0],1))
    x = y
    # z = np.dot(w.T,x) + b
    # a = np.tanh(z)
    #print("X shape " + str(x.shape))
    Z,A,A_final = multiLayerNeuralNetwork.forward_propogation(x,W,B)
    #print("As shape " + str(A_final.shape))
    a = A_final[0][0]
    #print(a)
    if (a > 0.5):
        return True
    else:
        return False


def predictCatFiles(W,B):
    catFileLst = "/Users/rasrivastava/neural_net/code/files/train/train/catFile100.txt"
    total = 0.0
    success = 0.0
    with open(catFileLst) as f:
        for catImageFile in f:
            catImageFile = catImageFile.replace("\n","")
            catImageFile = PREFIX + catImageFile
            total = total + 1.0
            if (predict(W,B,catImageFile) == True):
                success = success + 1.0

    print(success/total)

def predictDogFiles(W,B):
    dogFileLst = "/Users/rasrivastava/neural_net/code/files/train/train/dogFile100.txt"
    total = 0.0
    success = 0.0
    with open(dogFileLst) as f:
        for dogImageFile in f:
            dogImageFile = dogImageFile.replace("\n","")
            dogImageFile = PREFIX + dogImageFile
            total = total + 1.0
            if (predict(W,B,dogImageFile) == True):
                success = success + 1.0

    print(success/total)


def train_neural_network():
    X,Y = formInputOutput()
    print(X.shape)
    print(Y.shape)
    #X,Y = formInputOutputRandom()
    #print(X.shape)
    #print(Y.shape)
    print("Formatted successfully")
    W,B = multiLayerNeuralNetwork.nn_model(X,Y,2,5)
    predictCatFiles(W,B)
    print("for dogs")
    predictDogFiles(W,B)

    #print(W)
    #print(B)

train_neural_network()

