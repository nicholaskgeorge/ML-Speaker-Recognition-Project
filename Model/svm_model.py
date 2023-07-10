import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import librosa
import os

data_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_training_data.npy"
label_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_training_labels.npy"
mean_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_data_feature_mean.npy"
data = np.load(data_path)
labels = np.load(label_path)
mean = np.load(mean_path)

X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size=0.35, random_state=90)


# Create an SVM classifier
clf = svm.SVC(kernel='rbf')

# Train the SVM classifier
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)


# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)

# Get detailed statistics on specific speakers

speakers = np.unique(y_test)

for s in speakers:
    print(f'Statistics for speaker {s}:')
    speaker_data = X_test[y_test == s]
    print(f"The number of data points for test is {speaker_data.shape[0]}")
    labels = np.array([s]*speaker_data.shape[0])
    print(f"Speaker {s} is {labels.shape[0]/X_test.shape[0]} of the test set")
    pred = clf.predict(speaker_data)
    print(f"Accuracy score of {accuracy_score(labels, pred)}")

#Test on laptop mic data
#get names of all files in data folder
test_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\Test_audio_pat"
files = []
for file_name in os.listdir(test_path):
    if os.path.isfile(os.path.join(test_path, file_name)):
        files.append(file_name)


num_correct = 0
#load each one
for file in files:
    #get all mfcc data
    full_data_path = os.path.join(test_path, file)
    signal, sr = librosa.load(full_data_path)
    signal = librosa.util.normalize(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).flatten().reshape(1,-1)-mean
    if int(clf.predict(mfcc)[0])==6:
        num_correct+=1

print(f'Laptop accuracy was {num_correct/len(files)}.')

filename = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Model\svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))