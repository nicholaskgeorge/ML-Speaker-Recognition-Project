import librosa
import os
from pydub import AudioSegment
from math import ceil
import numpy as np

raw_data_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\raw_speaker_data\v3"
dest_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\all_speakers"

slice_len = 2
min_audio_len = 1
sample_num = 0
#loop through folder with files

#get names of all files in data folder
files = []
for file_name in os.listdir(raw_data_file_path):
    if os.path.isfile(os.path.join(raw_data_file_path, file_name)):
        files.append(file_name)

speaker = int(files[0][1:files[0].find("_")])

for file in files:

    audio = AudioSegment.from_file(os.path.join(raw_data_file_path, file))

    # Calculate the duration of the audio
    audio_duration = len(audio)

    start = 0
    end = slice_len*1000

    for i in range(ceil(audio_duration/(slice_len*1000))):
        # Check if the audio can be evenly split into two segments
        if audio_duration-end >= min_audio_len:

            # Split the audio into two segments
            new_audio = audio[start:end]

            file_name = f"v{speaker}_split_s{sample_num}.wav"

            new_audio.export(os.path.join(dest_file_path, file_name), format='wav')

            start += slice_len*1000
            end += slice_len*1000
            sample_num += 1
        
        # else:

        #     # Split the audio into two segments
        #     new_audio = audio[start:end]  # Convert to milliseconds
        #     file_name = f"v{speaker}_split_s{sample_num}.wav"

        #     new_file_path = os.path.join(dest_file_path, file_name)

        #     # Export the spliced audio segments to separate files
        #     new_audio.export(new_file_path, format='wav')
            
        #     audio = AudioSegment.from_file(new_file_path)

        #     # Calculate the duration of the audio
        #     audio_duration = len(audio) / 1000  # Convert from milliseconds to seconds

        #     # Calculate the amount of padding required in milliseconds
        #     padding_duration = (slice_len - audio_duration+1) * 1000

        #     # Perform zero padding
        #     padded_audio = AudioSegment.silent(duration=padding_duration)

        #     # Concatenate the original audio and padded audio
        #     output_audio = audio + padded_audio

        #     # Export the zero-padded audio to a file
        #     output_audio.export(new_file_path, format='wav')
