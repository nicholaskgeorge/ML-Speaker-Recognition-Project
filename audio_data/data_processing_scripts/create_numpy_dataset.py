import librosa
import numpy as np
import os 

def get_mffcc_stuff(data_path):
    # load audio files with librosa
    signal, sr = librosa.load(data_path)

    #normalize the audio
    signal  = librosa.util.normalize(signal)
    #get mfccs and all deltas
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    # delta_mfccs = librosa.feature.delta(mfccs)
    # delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    #make into one data point
    mfccs_features = mfccs #np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    #flatten data set
    data_point = mfccs_features.flatten()

    return data_point

raw_data_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\augmented_data"
dest_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset"

#loop through folder with files

#get names of all files in data folder
files = []
for file_name in os.listdir(raw_data_file_path):
    if os.path.isfile(os.path.join(raw_data_file_path, file_name)):
        files.append(file_name)

#get the number of features in the data set
full_data_path = os.path.join(raw_data_file_path, files[0])
feature_num = len(get_mffcc_stuff(full_data_path))


#initiate matrix
data_matrix = np.empty((0, feature_num))
label_vector = np.array([])

#load each one
for file in files:
    #get all mfcc data
    full_data_path = os.path.join(raw_data_file_path, file)
    data_point = get_mffcc_stuff(full_data_path)

    #add the vector to matrix
    data_matrix = np.vstack((data_matrix, data_point))

    #add to label vector
    label = int(file[1:file.find("_")])
    label_vector = np.append(label_vector, label)

# Do normalization
data_mean = data_matrix.mean(axis=0)
data_matrix = data_matrix-data_mean

# shuffle the data
np.random.seed(49)


# Shuffle the indices
shuffled_indices = np.random.permutation(len(data_matrix))

# # Shuffle the dataset and labels together
# data_matrix = data_matrix[shuffled_indices]
# label_vector = label_vector[shuffled_indices]

#save the matrix
training_data_path = os.path.join(dest_file_path, "speaker_training_data.npy")
training_label_path = os.path.join(dest_file_path, "speaker_training_labels.npy")
data_mean_path = os.path.join(dest_file_path, "speaker_data_feature_mean.npy")

np.save(training_data_path, data_matrix)
np.save(training_label_path, label_vector)
np.save(data_mean_path, data_mean)



