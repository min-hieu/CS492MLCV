import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from time import time
from tqdm import tqdm 

######## for styling ########
from matplotlib import font_manager, rcParams
font_manager.findSystemFonts(fontpaths="/Users/charlie/Library/Fonts/", fontext="ttf")
myfont = {'fontname':'Iosevka', 'fontsize':'15', 'fontweight': 'bold'}
#############################

########### Setup ###########
dataset = scipy.io.loadmat('face.mat')
data    = np.array(dataset['X'])     # (2576, 520) 
labels  = np.array(dataset['l'][0])  # (1, 520)


test_indices  = np.concatenate((np.arange(data.shape[1]//10)*10+7, 
                               (np.arange(data.shape[1]//10)*10+8)))
train_indices = [i for i in range(data.shape[1]) if i not in test_indices]

train_data  = data[:, train_indices]
train_label = labels[train_indices]

test_data  = data[:, test_indices]
test_label = labels[test_indices]

def show_face(x, title=None):
    '''
    Input: 
        x: (2576,) vector representing face
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    plt.title(title, **myfont, pad=13)
    face = x.reshape((46, 56)).T
    plt.imshow (face,cmap='gray')

    filename = "../figures/result.png"
    if title:
        filename = f"../figures/{title}.png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show ()

def plot_acc(acc, title=None):
    '''
    Input: 
        x: (2576,) vector representing face
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    plt.title(title, **myfont, pad=13)
    x = range(len(acc))
    plt.plot(x, acc)

    filename = "../figures/acc_result.png"
    if title:
        filename = f"../figures/{title}.png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show ()
#############################

############ Q1 #############
def time_func(name, func, arg):
    start = time()
    result = func(arg)
    print(f"{name} took {time()-start}")
    return result

def face_recon(i, eival, eivec):
    '''
    Input: 
        X: (2576, N) matrix with face vector along 
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    M = [10, 50, 100, 200, 300, 400]
    pass

def train():
    mean = train_data.mean(axis=1)
    A = (train_data.T - mean.T).T
    N = A.shape[1]

    eival_slow, eivec_slow = time_func("slow method", np.linalg.eigh, A @ A.T / N)
    eival_fast, eivec_fast = time_func("fast method", np.linalg.eigh, A.T @ A / N)
    print("number of non-zero eigen values slow: ", np.sum(eival_slow > 1e-8))
    print("number of non-zero eigen values fast: ", np.sum(eival_fast > 1e-8))
    print(f"eigen value are same? {(eival_slow[-416:]-eival_fast < 1e-8).all()}")

    print(np.min(eivec_slow))
    print(np.min(A @ eivec_fast))
    return mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast

# mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast = train()
'''
show_face(mean, "The Mean Face")
for i in range(1, 4):
    show_face(eivec_slow[:, -i], f"{i} Eigenface (slow)")
    show_face(A @ eivec_fast[:, -i], f"{i} Eigenface (fast)")
'''
def reconstruct_face (mean, phi_w, V):
    face = mean.copy()
    for i, eig in enumerate(V.T):
        face += eig * phi_w[i]
    return face

N_nonzero = np.sum(eival_slow > 1e-8)
def test():
    acc_list = []
    sample = 42
    for M in tqdm(range(1, N_nonzero+1), total=N_nonzero):
        V = eivec_slow[:, -M:] # eigen space consist of top M eigen vectors
        W = V.T @ A            # column vectors are w_n 

        # testing 
        test_data  = data[:, test_indices]
        test_label = labels[test_indices]

        acc = 0
        test_N = test_data.shape[1]
        for i, test_face in enumerate(test_data.T):
            phi   = test_face - mean 
            phi_w = (V.T @ phi) # project into eigenspace
            nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

            pred_label = train_label[nn]
            recon_face = reconstruct_face(mean, phi_w, V)
            if (i == sample and M % 50 == 0):
                show_face(test_face, f"original face, M={M}")
                show_face(phi, f"normalized face, M={M}")
                show_face(recon_face, f"reconstucted face, M={M}")
                show_face(train_data[:, nn], f"closest face in eigenspace, M={M}")
            acc += (pred_label == test_label[i])

        acc_list.append(acc/test_N)

    plot_acc(acc_list, "Accuracy vs. M")

# test()
#############################

############ Q2 #############

# split training subsets
subset1_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+0, (np.arange(train_data.shape[1]//8)*8+1)))
train_subdata_1  = train_data[subset1_indices]
train_sublabel_1 = train_label[subset1_indices]

subset2_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+2, (np.arange(train_data.shape[1]//8)*8+3)))
train_subdata_2  = train_data[subset2_indices]
train_sublabel_2 = train_label[subset2_indices]

subset1_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+4, (np.arange(train_data.shape[1]//8)*8+5)))
train_subdata_3  = train_data[subset3_indices]
train_sublabel_3 = train_label[subset3_indices]

subset1_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+6, (np.arange(train_data.shape[1]//8)*8+7)))
train_subdata_4  = train_data[subset4_indices]
train_sublabel_4 = train_label[subset4_indices]

def recon_error (mean, V, X):
    N   = X.shape[1] # x_n is column of X
    err = zeros_like(mean)
    for i, face in enumerate(X.T):
        phi_w = V.T @ (face - mean) 
        recon_face = reconstruct_face(mean, phi_w, V)
        err += np.linalg.norm(face - recon_face)
    return err / N

def face_reg_acc (A, V, test_data, test_label, title=None):
    W = V.T @ A            # column vectors are w_n 

    acc = 0
    test_N = test_data.shape[1]
    for i, test_face in enumerate(test_data.T):
        phi   = test_face - mean 
        phi_w = (V.T @ phi) # project into eigenspace
        nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

        pred_label = train_label[nn]
        recon_face = reconstruct_face(mean, phi_w, V)
        acc += (pred_label == test_label[i])
    acc /= tet_N
    print(f"{title} face recognition accuracy: {acc}")
    return acc

def batch_PCA():
    pass

def first_set_PCA():
    pass

def increment_PCA():
    pass

#############################

############ Q3 #############
# TODO
#############################

############ Q4 #############
# TODO
#############################
