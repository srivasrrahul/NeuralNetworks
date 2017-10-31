#X -- (input_size,examples)
#Y -- (output_size,examples)


import numpy as np

def initialize_layers(X,Y,hidden_layers):
    n_x = X.shape[0]
    n_h = {}

    for i in range(0,hidden_layers):
        n_h[str(i+1)] = 5

    n_y = Y.shape[0]
    return (n_x, n_h, n_y)



def initialize_parameters(n_x,n_h,n_y):
    W = {}
    #W["0"] = n_x
    B = {}
    l = len(n_h)
    n_h["0"] = n_x
    for i in range(0, l):
        #print(i)
        #print("Ws index " + str(i+1))
        #W[str(i+1)] = np.random.randn(n_h[str(i)],n_h[str(i+1)])*0.01
        W[str(i+1)] = np.random.randn(n_h[str(i+1)],n_h[str(i)])*0.01
        #print("here")
        B[str(i+1)] = np.zeros((n_h[str(i+1)],1))
        #print(W)

    #W[str(l+1)] = np.random.randn(n_h[str(l-1)],n_y)*0.01
    W[str(l+1)] = np.random.randn(n_y,n_h[str(l-1)])*0.01
    #print("Ws index " + str(l+1))
    B[str(l+1)] = np.zeros((n_y,1))
    return (W,B)



def forward_propogation(X,W,B):
    Z = {}
    A = {}
    A[str(0)] = X
    A_final = None
    #print(range(0,len(W)))
    for i in range(0,len(W)):
        #print(i)
        #print("Zs index " + str(i+1))
        if (i != (len(W)-1)):
            #print("W shape " + str(W[str(i+1)].shape))
            #print("B shape " + str(B[str(i+1)].shape))
            Z[str(i+1)] = np.dot(W[str(i+1)],A[str(i)]) + B[str(i+1)]
            A[str(i+1)] = np.tanh(Z[str(i+1)])
            #A[str(i+1)] = 1.0/(1.0+np.exp(-Z[str(i+1)]))
        else:
            Z[str(i+1)] = np.dot(W[str(i+1)],A[str(i)]) + B[str(i+1)]
            A[str(i+1)] = 1.0/(1.0+np.exp(-Z[str(i+1)]))
            A_final = A[str(i+1)]
    return (Z,A,A_final)


def cost_function(A_final,Y):
    #logprobs = np.multiply(np.log(A_final),Y)
    m = Y.shape[1]
    log1 = (1.0/m)*(np.dot(np.log(A_final),Y.T))
    log2 = (1.0/m)*(np.dot(1.0-Y,np.log(1.0-A_final).T))
    logprobs = log1 + log2
    cost = -logprobs
    cost = np.squeeze(cost)
    return cost


def backward_propogation(W,B,X,Y,A,A_final):
    m = X.shape[1]
    dW = {}
    dB = {}
    dZ = A_final - Y
    #print(Y)
    for i in range(len(W),0,-1):
        #print("i is " + str(i))
        #print("dz size " + str(dZ.shape))
        #print("w size " + str(W[str(i)].shape))
        #print("back propogation " + str(i))

        #dW[str(i)] = (1/m)*np.dot(A[str(i-1)],dZ.T)
        dW[str(i)] = (1.0/m)*(np.dot(dZ,A[str(i-1)].T))
        #print("dw size " + str(dW[str(i)].shape))
        #dB[str(i)] = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dB[str(i)] = (1.0/m)*(np.sum(dZ,axis=1,keepdims=True))
        #dZ = np.multiply(np.dot(W[str(i)],dZ),(1-np.power(A[str(i-1)],2)))
        #temp = np.dot(W[str(i)].T,dZ)
        dZ = np.multiply(np.dot(W[str(i)].T,dZ),(1-np.power(A[str(i-1)],2)))
    return dW,dB



def update_parameters(W,B,dW,dB,learning_rate):
    l = len(W)
    W_updated = {}
    B_updated = {}
    for i in range(1,l+1):
        index = str(i)
        w = W[index]
        b = B[index]
        dw = dW[index]
        db = dB[index]
        W_updated[index] = w - learning_rate*dw
        B_updated[index] = b - learning_rate*db
    return W_updated,B_updated


def nn_model(X,Y,hidden_layers,num_iterations) :
    n_x,n_h,n_y = initialize_layers(X,Y,hidden_layers)
    #print("Layer sizes " + str(n_x) + " " + str(n_y))
    W,B = initialize_parameters(n_x,n_h,n_y)
    # for w in W.keys():
    #     print("key " + str(w) + str(W[w].shape))
    for i in range(0,num_iterations):
        Z,A,A_final = forward_propogation(X,W,B)
        cost = cost_function(A_final,Y)
        dW,dB = backward_propogation(W,B,X,Y,A,A_final)
        #print("Updated dW " + str(dW))
        W,B = update_parameters(W,B,dW,dB,0.20)
        print("Cost is " + str(cost))

    return W,B







# X = np.random.randn(4,10)
# Y = np.random.randn(1,10)
# # (n_x,n_h,n_y) = initialize_layers(X,Y,2)
# # #print(n_h)
# # (W,B) = initialize_parameters(n_x,n_h,n_y)
# # for k in W.keys():
# #     print(str(k) + " " + str(W[k].shape))
# #     print(str(k) + " " + str(B[k].shape))
#
#
#
# W,B = nn_model(X,Y,2,5)
#
    
