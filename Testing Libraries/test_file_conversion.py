import ffmpeg


from pydub import AudioSegment

# Load the audio file using pydub
audio_file = AudioSegment.from_file("speaker1_Nick_test.m4a", format="m4a")

# Export the audio file as a WAV file
audio_file.export("output_file.wav", format="wav")