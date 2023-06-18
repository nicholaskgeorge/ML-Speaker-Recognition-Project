from pydub import AudioSegment


"""
Goals:
add new stuff is easy
configure minimum length of songs,
split songs into chunks padd the rest
padding length
"""
def splice_audio(input_file, output_file1, output_file2, splice_duration):
    audio = AudioSegment.from_file(input_file)
    

    # Split the audio into two segments
    audio1 = audio[:splice_duration * 1000]  # Convert to milliseconds
    audio2 = audio[splice_duration * 1000:]

    # Export the spliced audio segments to separate files
    audio1.export(output_file1, format='wav')
    audio2.export(output_file2, format='wav')

# Example usage: splice input.wav into two 2-second segments: output1.wav and output2.wav
splice_audio('song.wav', 'output1.wav', 'output2.wav', 2)