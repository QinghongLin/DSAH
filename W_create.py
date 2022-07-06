import numpy as np
import scipy.io as scio

def L2_dist(X, Y):
    shape = X.shape
    xx = np.sum(np.multiply(X, X), 1).reshape([shape[0], 1])
    yy = np.sum(np.multiply(Y, Y), 1).reshape([shape[0], 1])
    xy = np.matmul(X, Y.T)
    D = xx + yy.T - 2 * xy
    D[D < 0] = 0
    return D

def train_set(dataset):
    assert dataset in ['cifar', 'flickr', 'nus']
    if dataset == 'cifar':
        train_feature = scio.loadmat('/public/dataset/cifar-10/cifar_train_vgg19_relu7.mat')['train_data']
    elif dataset == 'flickr':
        train_feature = scio.loadmat('/public/dataset/flickr25k/flickr_train_vgg19_relu7.mat')['train_data']
    elif dataset == 'nus':
        train_feature = scio.loadmat('/public/dataset/nus-wide/nus_train_vgg19_relu7.mat')['train_data']
    return train_feature

def W(feature, K1=20, K2=30):
    dist = L2_dist(feature, feature)

    data = np.argsort(dist, 1)[:,:1000]
    size = data.shape[0]
    S = np.ones((size, size)) * -1
    for i in range(size):
        top = set(data[i][:K1 + 1])  # exclude ifself so add one
        nums = []
        idx = []
        for j in range(size):
            print('(' + str(i) + ',' + str(j) + ')' + '/' + '(' + str(size) + ',' + str(size) + ')')
            si = set(data[j][:K1 + 1])
            both = si & top
            nums.append(len(both))
            idx.append(j)
        nums = np.array(nums)
        indx = np.argsort(nums * -1)[:K2]
        sp = set()
        for ii in indx:
            idx_ = idx[ii]
            # sp = sp | set(data[idx_][:K1 + 1])    # bgan
            sp = sp & set(data[idx_][:K1 + 1])
        for j in sp:
            S[i][j] = 1.
            S[j][i] = 1.
    return S

def save(dataset, S, name):
    assert dataset in ['cifar', 'flickr', 'nus']
    if dataset == 'cifar':
        scio.savemat('cifar-10/' + name + '.mat', {'S':S})
    elif dataset == 'flickr':
        scio.savemat('flickr25k/' + name + '.mat', {'S':S})
    elif dataset == 'nus':
        scio.savemat('nus-wide/' + name + '.mat', {'S':S})

if __name__ == '__main__':
    # dataset = 'cifar'
    for dataset in ['flickr']:
        feature = train_set(dataset)
        S = W(feature, K1=20, K2=30)
        save(dataset, S, 'S_K1_20_K2_30')
        print("Save done!")