import os
from pydub import AudioSegment

def convert_to_wav(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Filter the files to only include .m4a files
    m4a_files = [file for file in files if file.endswith('.m4a')]

    for m4a_file in m4a_files:
        # Construct the input and output file paths
        input_path = os.path.join(input_folder, m4a_file)
        output_path = os.path.join(output_folder, os.path.splitext(m4a_file)[0] + '.wav')

        # Load the .m4a file
        audio = AudioSegment.from_file(input_path, format='m4a')

        # Export the audio as .wav file
        audio.export(output_path, format='wav')

        print(f"Converted: {input_path} -> {output_path}")

# Example usage
input_folder = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\raw_speaker_data\v6(Pat)'
output_folder = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\raw_speaker_data\v6(Pat)'
convert_to_wav(input_folder, output_folder)
