    # #amplitude flipping
    # if random()<=amplitude_flipping_prob:
    #     file_name = f"v{speaker}_s{sample_num}.wav"
    #     sample_num+=1
    #     flipped_audio = audio*(-1)
    #     sf.write(os.path.join(dest_file_path, file_name), flipped_audio, sr)