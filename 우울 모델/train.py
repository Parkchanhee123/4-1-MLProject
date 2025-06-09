import torch
from models import MLP

model = MLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    x = torch.randn(10, 100)
    y = torch.randint(0, 2, (10,))
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch", epoch, "Loss:", loss.item())
