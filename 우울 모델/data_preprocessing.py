import soundfile as sf
import numpy as np

def preprocess_wav(input_path, output_path):
    signal, fs = sf.read(input_path)
    signal = signal / np.max(np.abs(signal))
    sf.write(output_path, signal, fs)

if __name__ == '__main__':
    preprocess_wav("sample.wav", "output.wav")
