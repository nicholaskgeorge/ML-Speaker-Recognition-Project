import numpy as np
import os 
from pydub import AudioSegment

raw_data_file_path = "C:/Users/nicok/Documents/ML-Speaker-Recognition-Project/downloaded_audio/LibriSpeech/dev-clean/174/168635"
dest_file_path = "C:/Users/nicok/Documents/ML-Speaker-Recognition-Project/audio_data/raw_speaker_data/v3"

def convert_flac_to_wav(flac_file_path, wav_file_path):
    # Load the FLAC file
    audio = AudioSegment.from_file(flac_file_path, format="flac")

    # Export as WAV file
    audio.export(wav_file_path, format="wav")

#loop through folder with files
speaker_num = 3
count = 20

#get names of all files in data folder
files = []
for file_name in os.listdir(raw_data_file_path):
    if os.path.isfile(os.path.join(raw_data_file_path, file_name)):
        files.append(file_name)

for file in files:
    name = f"v{speaker_num}_s{count}.wav"
    new_file = os.path.join(dest_file_path, name)
    to_convert = os.path.join(raw_data_file_path, file)
    convert_flac_to_wav(to_convert,new_file)
    count += 1
    



