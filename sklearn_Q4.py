import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='./')
import matplotlib.pyplot as plt
images = mnist.data
targets = mnist.target
X = mnist.data / 255.
Y = mnist.target
img1 = X[0].reshape(28, 28)
plt.imshow(img1, cmap='gray')
img2 = 1 - img1
plt.imshow(img2, cmap='gray')
img3 = img1.transpose()
plt.imshow(img3, cmap='gray')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
from sklearn.svm import LinearSVC
from sklearn import metrics
Cs=[0.018]
train_scores=[]
test_scores=[]
for C in Cs:
    svc=LinearSVC(C=C)
    svc.fit(X_train,Y_train)
    train_scores.append(svc.score(X_train,Y_train))
    test_scores.append(svc.score(X_test, Y_test))

    ## 绘图
print(Cs)	
print(train_scores)
print(test_scores)

print('Training Score: %.2f' % svc.score(X_train,Y_train))
print('Testing Score: %.2f' % svc.score(X_test, Y_test))
train_accuracy=svc.score(X_train,Y_train)
test_accuracy=svc.score(X_test,Y_test)
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

