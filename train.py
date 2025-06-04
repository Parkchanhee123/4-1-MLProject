import torch
import numpy as np
import soundfile as sf
from models import MLP

model = MLP(input_dim=16000, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_list = ["test1.wav", "test2.wav"] 
label_dict = {"test1.wav": 0, "test2.wav": 1} 

for epoch in range(2):
    for fname in train_list:
        signal, sr = sf.read("data/" + fname)
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = signal[:16000]
        signal = signal / np.max(np.abs(signal))

        x = torch.tensor(signal).float().unsqueeze(0)
        y = torch.tensor([label_dict[fname]]).long()

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}, loss: {loss.item():.4f}")
