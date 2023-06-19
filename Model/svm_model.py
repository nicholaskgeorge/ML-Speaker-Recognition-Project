import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_training_data.npy"
label_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_training_labels.npy"
data = np.load(data_path)
labels = np.load(label_path)

X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, random_state=9)



# Create an SVM classifier
clf = svm.SVC(kernel='rbf', degree = 6)

# Train the SVM classifier
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)