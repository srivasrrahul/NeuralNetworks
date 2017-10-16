import numpy as np
import PIL
import PIL.Image
import cv2


PREFIX = "/Users/rasrivastava/neural_net/code/files/train/train/"

PERCENT = 20
IMAGE_SIZE = (15,100)
def findMaxHeightWidth(catFileName):
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
                x = updatedArr.flatten()
                x = x/255.0
                if (assigned == False):
                    X = x
                    assigned = True
                else:
                    X = np.column_stack((X,x))
            else:
                #print("Skip sample " + str(count))
                count = 0

    return X


def getDogFileName(dogFileName):
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
    X1 = findMaxHeightWidth("/Users/rasrivastava/neural_net/code/files/train/train/catFile.txt")
    X2 = getDogFileName("/Users/rasrivastava/neural_net/code/files/train/train/dogFile.txt")

    Y1 = np.ones((1,X1.shape[1]))
    Y2 = np.zeros((1,X2.shape[1]))

    X = np.concatenate((X1,X2),axis=1)

    Y = np.concatenate((Y1,Y2),axis=1)
    return ((X*1.0),Y*1.0)



def initialValue(X):
    #w = np.ones((X.shape[0],1))*0.1
    w = np.random.rand(X.shape[0],1)
    b = 0.0
    return w,b

def propogate(w,b,X,Y):

    #print("w is " )
    #print(w)
    
    m = X.shape[1] #of samples
    m = m * 1.0
    Z = (np.dot(w.T,X)  + b)
    #print("Z is ")
    #print(Z)
    #A = 1/(1+np.exp(-Z))
    A = np.tanh(Z)
    #print(Z.shape)
    #print("A is ")
    #print(A)
    cost = (1/m)*(-1)*(np.dot(Y,np.log(A.T)) + np.dot((1-Y),np.log(1-A.T))) 
    #print(Z)
    
    cost = np.squeeze(cost)
    dz = A - Y
    #print("DZ is ") 
    #print(dz)
    dw =  (1/m)*(np.dot(X,dz.T))
    db = (1/m)*(np.sum(Z))

    return ((dw,db),cost)



def optimize(w,b,X,Y,num_iterations,learning_rate):
    costs = []
    for i in range(num_iterations):
        ((dw,db),cost) = propogate(w,b,X,Y)
        if (i % 100 == 0):
            print("DW  " + str(dw))
            print("DB " + str(db))
            print("Cost updated is " + str(cost))
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
    return (w,b)


def predict(w,b,imageFile):
    arr = np.asarray(PIL.Image.open(imageFile))
    updatedArr = cv2.resize(arr, IMAGE_SIZE) 
    x = updatedArr.flatten() 
    x = x/255.0
    z = np.dot(w.T,x) + b
    a = np.tanh(z)
    if (a > 0.5):
        return True
    else:
        return False

def predictCatFiles(w,b):
    catFileLst = "/Users/rasrivastava/neural_net/code/files/train/train/catFile100.txt"
    total = 0.0
    success = 0.0
    with open(catFileLst) as f:
        for catImageFile in f:
            catImageFile = catImageFile.replace("\n","")
            catImageFile = PREFIX + catImageFile
            total = total + 1.0
            if (predict(w,b,catImageFile) == True):
                success = success + 1.0
            
    print(success/total)
            
        
    
    

#retValue = findMaxHeightWidth("./train/catFile.txt")
#print(retValue)
X,Y = formInputOutput()
#print(X.shape)
#print(X)
# print("X is ")
# print(X)
# print(X > 1.0).all()
w,b = initialValue(X)
w,b = optimize(w,b,X,Y,1000,0.1)
print("Weight is ")
print(w)
print(b)
predictCatFiles(w,b)
