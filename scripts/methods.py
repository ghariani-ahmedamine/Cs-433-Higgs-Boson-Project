import numpy as np 
from proj1_helpers import predict_labels

#############################Data Transformation###############################
def standardize(tX):
    """
    Calculate the Mean Square Error of the given paramaters

    Parameters
    ----------
    tX : np.array
        Array of the features (N,D)


    Returns:
        std_data (N,D) the standardized features Array  
    """
    centered_data = tX - np.mean(tX, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

def missing_to_value(tX , value):
    """
    Replace missing values with chosen value

    Parameters
    ----------
    tX : np.array
        Array of the features (N,D)


    Returns:
        tX after replacement of missing values  
    """    
    for i in range(tX.shape[0]):
        for j in range(tX.shape[1]):
            if(tX[i,j]==-999):
                tX[i,j]=value
    return tX

def nan_to_median(tX):
    """
    Replace remaining missing values in each coloumn by the median of the feature 
    
    Parameters
    ----------
    tX : np.array
        Array of the features (N,D)


    Returns:
        tX after replacement of missing values 
    """
    median_feature=np.nanmedian(tX,axis=0)
    for i in range(tX.shape[0]):
        for j in range(len(median_feature)):
            if(np.isnan(tX[i,j])):
                tX[i,j]=median_feature[j]
    return tX

def PCA(X , num_components):
     
    #mean of X
    X_mean = X - np.mean(X , axis = 0)
     
    #covariance matrix
    cov_mat = np.cov(X_mean , rowvar = False)
     
    #calculate eigenvalues,vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #sort eigenvectors by eigenvalue
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Select component number
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #reduce dataset
    X_reduced = np.dot(eigenvector_subset.transpose() , X_mean.transpose() ).transpose()
     
    return X_reduced

#############################Cross-validation##################################
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def accuracy_score(y , tX , w):
    """
    Calculate the accuracy score 

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features (N,D)
    w : np.array
        Array of weights (D,)

    Returns:
        The accuracy score
    """
    y_pred = predict_labels(w , tX)
    return np.sum(y_pred==y) / len(y)

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx,k, k_indices, method,lambda_,degree):
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = tx[te_indice,:]
    x_tr = tx[tr_indice,:]
    if (degree != 0):
        x_tr = build_poly_dataset(x_tr,degree)
        x_te = build_poly_dataset(x_te,degree)
        
    # methods
    if method == 'least_squares_GD':
        w , _ = least_squares_GD(y_tr, x_tr, np.zeros(x_tr.shape[1]),500 ,0.5)
    if method == 'least_squares_SGD':
        w , _ = least_squares_SGD(y_tr, x_tr, np.zeros(x_tr.shape[1]),50 ,500 ,0.5)
    if method == 'least_squares':
        w , _ = least_squares(y_tr, x_tr)
    if method == 'ridge_regression':
        w , _ = ridge_regression(y_tr, x_tr,lambda_)
    if method == 'logistic_regression':
        w , _ = logistic_regression(y_tr,x_tr,np.zeros(x_tr.shape[1]),500,0.5)
    if method == 'reg_logistic_regression':
        w , _ = reg_logistic_regression(y_tr, x_tr,lambda_,np.random.rand(x_tr.shape[1]),500,0.5)

    
    # calculate the loss for train and test data
    
    loss_tr = loss_MSE(y_tr, x_tr, w)
    loss_te = loss_MSE(y_te, x_te, w)
    accuracy = accuracy_score(y_te,x_te,w)
    return loss_tr, loss_te,accuracy,w

def cross_validation_average_accuracy(y,x,k_fold,seed,method,lambda_,degree):
     
    
        # split data in k fold
        k_indices = build_k_indices(y, k_fold, seed)
        # define lists to store the loss of training data and test data
        accuracy_tmp = []
        ws = []
        # cross validation
        for k in range(k_fold):
            loss_tr, loss_te,accuracy,w = cross_validation(y, x,k, k_indices, method,lambda_,degree)
            accuracy_tmp.append(accuracy)
                   
        return np.mean(accuracy_tmp)
def cross_validation_demo_least_squares_PCA(y,x,k_fold,seed,method):
     
    iter_ = []
    accuracy_=[]
    for i in range (x.shape[1]):
        iter_.append(i)          
        tX = PCA(x,i)
        # split data in k fold
        k_indices = build_k_indices(y, k_fold, seed)
        # define lists to store the loss of training data and test data
        accuracy_tmp = []
        # cross validation
        for k in range(k_fold):
            loss_tr, loss_te,accuracy,_ = cross_validation(y, tX,k, k_indices, method,0,0)
            accuracy_tmp.append(accuracy)
        accuracy_.append(np.mean(accuracy_tmp))            
    plt.scatter(iter_,accuracy_)  
    plt.xlabel('number of principal components')
    plt.ylabel('accuracy')
    plt.title('Accuracy variation in respect to number of principal components')

def cross_validation_demo_lambdas(y,x,k_fold,seed,method,lambdas,degree):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    accuracy_ = []
    # cross validation
    for lambda_ in lambdas:
        mse_tr_tmp = []
        mse_te_tmp = []
        accuracy_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te,accuracy,w = cross_validation(y, x,k, k_indices, method,lambda_,degree)
            mse_tr_tmp.append(loss_tr)
            mse_te_tmp.append(loss_te)
            accuracy_tmp.append(accuracy)
        mse_tr.append(np.mean(mse_tr_tmp))
        mse_te.append(np.mean(mse_te_tmp))
        accuracy_.append(np.mean(accuracy_tmp))
    return mse_tr, mse_te,accuracy_,lambdas

def cross_validation_demo_degrees(y,x,k_fold,seed,method,lambda_,degrees):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    accuracy_ = []
    # cross validation
    for degree in degrees:
        mse_tr_tmp = []
        mse_te_tmp = []
        accuracy_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te,accuracy,w = cross_validation(y, x,k, k_indices, method,lambda_,degree)
            mse_tr_tmp.append(loss_tr)
            mse_te_tmp.append(loss_te)
            accuracy_tmp.append(accuracy)
        mse_tr.append(np.mean(mse_tr_tmp))
        mse_te.append(np.mean(mse_te_tmp))
        accuracy_.append(np.mean(accuracy_tmp))
    return mse_tr, mse_te,accuracy_,degrees


    
###########################Models implementation###############################
def loss_MSE(y , tX , w):
    """
    Calculate the Mean Square Error

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features (N,D)
    w : np.array
        Array of weights (D,)

    Returns:
        The MSE loss 
    """
    e = y - tX @ w # (N,) #
    
    return (e.T @ e) / (2 * len(y))

def compute_gradient(y, tX , w):
    """
    Compute the gradient

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features (N,D)
    w : np.array
        Array of weights (D,)

    Returns:
        The gradient 
    """
    e = y - tX @ w # (N,) #
    
    return - (tX.T @ e) / len(y) , e

def least_squares_GD(y, tX, initial_w,max_iters, gamma):
    """
    Linear regression using gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Array of the initial weights of the model (D,)
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector along with its associated loss.

    """

    w = initial_w
    for n_iter in range(max_iters):
        gradient,err = compute_gradient(y, tX, w)
        w = w-gamma*gradient
        loss = loss_MSE(y, tX, w)
    return w,loss
       
        
    return w,loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    batch_size : int
        Number of samples of the batch
    num_batches: int
        The number of batches (default is 1)
    shuffle: bool
        Randomize the dataset (default is True)
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tX, initial_w, batch_size, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Array of  the initial  weights (D,)
    batch_size : int
        Number of samples of the batch
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w,loss) the last weight vector along with its associated loss.
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tX in batch_iter(y, tX, batch_size=batch_size,num_batches=1):
            
            gradient = compute_gradient(minibatch_y , minibatch_tX , w)
            w = w - gamma * gradient
            loss = loss_MSE(minibatch_y, minibatch_tX, w)

    return w,loss

def least_squares(y, tX):
    """
    Least squares regression using normal equations

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features  (N,D)

    Returns:
        (w, loss) the last weight vector along with its associated loss.

    """
    #Calculating w using normal equations
    
    w = np.linalg.solve(tX.T @ tX , tX.T @ y)
    loss = loss_MSE(y,tX,w)
    
    return w,loss

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_poly_dataset(tX,degree):
    tX_expanded = build_poly(tX[:,0] , degree)
    for i in range (tX.shape[1]-1):
        tX_expanded = np.concatenate((tX_expanded , build_poly(tX[:,i+1] , degree)),axis=1)
    return tX_expanded

def ridge_regression(y, tX, lambda_):
    """
    Calculate the Ridge Regression using normal equations

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tX : np.array
        Array of the features (N,D)
    lambda_ : np.float64
        Regularization parameter

    Returns:
        (w, loss) the last weight vector along with its associated loss.
    """
    
    lambda_prime = 2 * lambda_ *len(y)
    w= np.linalg.solve(tX.T @ tX + (lambda_prime * np.identity(tX.shape[1])) , tX.T @ y)
    loss=loss_MSE(y,tX,w)
    return w,loss

def sigmoid(t):
    """
    Compute the sigmoid

    Parameters
    ----------
    t : float
        
    Returns:
        The sigmoid of t
    """
    return 1/(1+np.exp(-t))


    """
    Logistic regression using gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    initial_w : np.array
        Initial random weights of the model (D,)
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).

    """
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        pred = sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(- loss)
        pred = sigmoid(tx.dot(w))
        gradient  = tx.T.dot(pred - y)
        w = w - gamma * gradient

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w.squeeze(),loss

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


    """
    Regularized logistic regression using gradient descent

    Parameters
    ----------
    y : np.array
        Array of labels (N,)
    tx : np.array
        Array of the features  (N,D)
    lambda_ : np.float64
        Regularization parameter
    initial_w : np.array
        Initial random weights of the model (D,)
    max_iters: int
        The maximum number of iterations
    gamma: float
        The step size

    Returns:
        (w, loss) the last weight vector of the calculation, and the corresponding loss value (cost function).

    """
def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
        # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.

        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w= w - gamma*gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w.squeeze(),loss
    
