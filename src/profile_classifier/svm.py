import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics


def fit_svm(X, ntarget):
    data = np.array(X)
    mtarget = np.asarray(ntarget)
    print(data, mtarget)
    print("saving arrays...")
    np.save('svm_data/Xdata', X)
    np.save('svm_data/targets', mtarget)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, mtarget, test_size=.7,
                                                        random_state=55)  # 70% training and 30% test
    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))



def predict(profile):
    print("loading SVM...")

    data = np.load('svm_data/Xdata.npy')
    targets = np.load('svm_data/targets.npy')
    print(data)
    print(targets)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=.7,
                                                        random_state=55)  # 70% training and 30% test
    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf')  # RBF Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)
    profile = np.asarray(profile)
    # Return the prediction for the profile.
    print(profile)
    return clf.predict(profile.reshape(1,-1))
