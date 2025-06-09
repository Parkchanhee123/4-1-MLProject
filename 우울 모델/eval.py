import torch
from models import MLP

model = MLP()
model.load_state_dict(torch.load("model.pth"))
model.eval()

x = torch.randn(1, 100)
out = model(x)
pred = torch.argmax(out, dim=1)
print("Predicted class:", pred.item())
