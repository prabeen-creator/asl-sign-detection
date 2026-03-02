#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# load your data
# expected format: [label, x0, y0, x1, y1, .... x20, y20]
df = pd.read_csv('sign_mnist_test.csv')

# 2. split into features (X) and labels (y)
X = df.iloc[:,1:].values # Landmark coordinates
y = df.iloc[:, 0].values # the letter labels

# 3. split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# 4. initialize and train the svm
# we use 'RBF' kernel for non-linear hand shapes and 'c=1.0' for regularization
clf = svm.SVC(kernel='rbf', gamma = 'scale', C = 1.0, probability = True)
clf.fit(X_train, y_train)

# 5. evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 6. save the model for later use in cpp
joblib.dump(clf, 'asl_svm_model.pkl')

