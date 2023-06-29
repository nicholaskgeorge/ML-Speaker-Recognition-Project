import os

def rename_files(folder):
    # Get a list of all files in the folder
    files = os.listdir(folder)

    for index, file in enumerate(files, start=1):
        # Construct the old and new file paths
        old_path = os.path.join(folder, file)
        new_file_name = f"v6_s{index}.wav"
        new_path = os.path.join(folder, new_file_name)

        # Rename the file
        os.rename(old_path, new_path)

        print(f"Renamed: {old_path} -> {new_path}")

# Example usage
folder = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\raw_speaker_data\v6(Pat)'  # Replace with the actual folder path
rename_files(folder)
