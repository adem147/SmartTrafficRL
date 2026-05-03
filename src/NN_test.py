import torch
import torch.nn as nn
import torch.optim as optim

# Generate training data
x = torch.linspace(-5, 5, 100).unsqueeze(1)   # shape: [100, 1]
y = 2 * x + 1 + 0.2 * torch.randn(x.size())   # add tiny noise

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # input=1, output=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 2000

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
test_value = torch.tensor([[4.0]])
prediction = model(test_value)
print("Prediction for x=4:", prediction.item())