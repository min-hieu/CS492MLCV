import numpy as np
import matplotlib.pyplot as plt
import scipy.io

dataset = scipy.io.loadmat('face.mat')
data    = np.array(dataset['X'])  # (2576, 520) 
labels  = np.array(dataset['l'][0])  # (1, 520)

test_indices  = np.concatenate((np.arange(data.shape[1]//10)*10+7, 
                               (np.arange(data.shape[1]//10)*10+8)))
train_indices = [i for i in range(data.shape[1]) if i not in test_indices]

train_data  = data[:, train_indices]
train_label = labels[train_indices]

test_data  = data[:, test_indices]
test_label = labels[test_indices]

def show_face(x):
    face = x.reshape((46, 56)).T
    plt.imshow (face)
    plt.show ()


# Q1
mean = np.expand_dims(np.mean(train_data, axis=1), axis=1)
A = train_data - np.expand_dims(np.mean(train_data, axis=1), axis=1)
N = A.shape[1]

S = A @ A.T / N
eig_values_slow, eig_vectors_slow = np.linalg.eigh(S)
eig_values_fast, eig_vectors_fast = np.linalg.eigh(A.T @ A / N)

show_face(mean)
show_face(eig_vectors_slow[:, -1])
show_face(A @ eig_vectors_fast[:, -1])
print("number of non-zero eigen values slow: ", np.sum(eig_values_slow > 1e-8))
print("number of non-zero eigen values fast: ", np.sum(eig_values_fast > 1e-8))
