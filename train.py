import os
import torch
import soundfile as sf
import numpy as np
from models import MLP
from data_io_train import get_train_list

model = MLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_list = get_train_list()
label_dict = np.load("labels.npy", allow_pickle=True).item()  # {"file1.wav": 0, ...}

for epoch in range(5):
    loss_sum = 0
    for fname in train_list:
        wav_path = os.path.join("data", fname)
        signal, _ = sf.read(wav_path)
        signal = signal / np.max(np.abs(signal))
        if signal.ndim > 1:
            signal = signal[:, 0]  # mono

        if len(signal) < 100:
            continue  # skip too short

        x = torch.tensor(signal[:100]).float().unsqueeze(0)
        y = torch.tensor([label_dict[fname]]).long()

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch}, Loss: {loss_sum / len(train_list):.4f}")

torch.save(model.state_dict(), "model.pth")
