from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

######################Convert Data functions#####################
def unpickle(file):
    with open(file, 'rb') as fo:
        d= pickle.load(fo, encoding='bytes')
    return d
def images_to_matrix(images, colors_dim):
    dim = ( len(images), int((np.shape(np.ravel(images[0])))[0]/colors_dim))
    A = np.empty(dim, dtype = np.uint8)
    i = 0
    for img in images:
        single_img_reshaped = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(single_img_reshaped.astype('uint8'))
        image = image.convert("L")
        img_array = np.ravel(np.array(image))
        A[i] = img_array
        i = i + 1
    return A
def get_data(file):
    my_dict = unpickle(file)
    labels = my_dict[b'labels']
    imgs = my_dict[b'data']
    return images_to_matrix(imgs, 3), labels
def get_combine_matrix(files, size):
    final_matrix = np.empty((0, size), dtype = np.uint8)
    final_labels = []
    for file in files:
        matrix, labels = get_data(file)
        final_labels += labels
        final_matrix = np.concatenate((final_matrix, matrix), axis=0)
    return final_matrix,final_labels
####################################################################
####################Classification functions########################
def KNN(X,train_labels,Z,true_labels,K):
    errors_for_k = np.empty(len(K))
    print (np.shape(X), np.shape(Z))
    for i in range(len(true_labels)) :
        distance_vector = np.linalg.norm(X-Z[i],axis = 1)
        j = 0
        for k in K:
            top_k = np.argpartition(distance_vector, k)[:k]
            labels = np.take(train_labels,top_k)
            infer_label = np.argmax(np.bincount(labels))
            if infer_label != true_labels[i]:
                errors_for_k[j] +=1
            j+=1
    return errors_for_k/len(true_labels)
####################################################################
###########################PCA functions############################
def PCA(A):
    avg = np.empty(train_matrix.shape[1])
    for col in range(np.shape(A)[1]):
        column = A[:, col]
        avg[col] = np.mean(column)
    return A-avg

S = [10,20,50,100,300,750,1000]
K = [5,10,15,20,50]
pic_in_row_dim = 1024
train_files = ["./cifar-10-batches-py/data_batch_1", "./cifar-10-batches-py/data_batch_2","./cifar-10-batches-py/data_batch_3",
               "./cifar-10-batches-py/data_batch_4", "./cifar-10-batches-py/data_batch_5"]
test_files = ["./cifar-10-batches-py/test_batch"]
if __name__== "__main__":
    test_matrix,test_labels = get_combine_matrix(test_files,pic_in_row_dim)
    train_matrix, train_labels = get_combine_matrix(train_files,pic_in_row_dim)
    train_matrix_centered = PCA(train_matrix)
    test_matrix_centered = PCA(test_matrix)
    U = np.linalg.svd(train_matrix_centered, full_matrices=False)[0]
    errors_with_pca = {}
    for s in S:
        Us = U[:s, :]
        X = np.transpose(np.dot(Us, np.transpose(train_matrix_centered)))
        Z = np.transpose(np.dot(Us, np.transpose(test_matrix_centered)))
        errors_with_pca[s] = KNN(X,train_labels, Z, test_labels, K)
    errors_without_pca = KNN(train_matrix, train_labels, test_matrix, test_labels,K)
    for s in errors_with_pca.keys():
        plot_without_PCA, = plt.plot(K, errors_without_pca, 'r--', label = 'Without PCA')
        plot_with_PCA, = plt.plot(K, errors_with_pca[s], 'bs-', label = 'With PCA')
        plt.title('errors for s = ' + str(s))
        plt.legend(handles=[plot_with_PCA , plot_without_PCA])
        plt.show()