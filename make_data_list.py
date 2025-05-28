import glob

wav_list = glob.glob("data/**/*.wav", recursive=True)
with open("train.txt", "w") as f:
    for wav in wav_list:
        f.write(wav + "\n")
