import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the MNIST data
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

# 2. Separate features and labels
X_train = train_df.drop('label', axis=1).values / 255.0  # Normalize pixels
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values / 255.0
y_test = test_df['label'].values

# 3. Train SVM (The Control)
print("Training SVM Baseline...")
clf = svm.SVC(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)

# 4. Evaluate and Generate Confusion Matrix (Week 1 Deliverable)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix: SVM Baseline (MNIST)")
plt.savefig('svm_confusion_matrix.png')
print("Saved svm_confusion_matrix.png")