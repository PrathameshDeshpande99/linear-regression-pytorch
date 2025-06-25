import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from pathlib import Path

# Device agnostic code.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
weight = 0.7
bias = 0.3

start = 0
end = 1
step =  0.02

X = torch.arange(start, end, step).unsqueeze(dim=1) # Unsqueeze to add an extra dimension for compatibility with nn.Linear

y= weight * X + bias


train_split = int(0.8*len(X))

X_train = X[:train_split]
y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]

# Plot
def plot_predictions(train_data = X_train,
                     test_data = X_test, 
                     train_label = y_train, 
                     test_label = y_test, 
                     predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data.cpu(), train_label.cpu(), c="b", s=4, label="Training data")
    plt.scatter(test_data.cpu(), test_label.cpu(), c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c="r", s=4, label="Predictions data")
    plt.legend()
    plt.show()


# Building model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)

model_0 = LinearRegressionModel()
model_0.to(device)
print(f"current params set randomly: {model_0.state_dict()}")

# Loss and Optimizer functions
loss_f = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.001)

# Device agnostic code for data
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training & Testing
torch.manual_seed(0) # Setting a fixed random seed for reproducibility

epochs = 5000

epoch_count = []
train_loss_value = []
test_loss_value = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_f(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_test_preds = model_0(X_test)
        test_loss = loss_f(y_test_preds, y_test)

    if epoch % 20 == 0:
        epoch_count.append(epoch)
        train_loss_value.append(loss)
        test_loss_value.append(test_loss)
        print(f"Epoch = {epoch} | Training Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())

# Data back to cpu
X_train = X_train.cpu()
y_train = y_train.cpu()
X_test = X_test.cpu()
y_test = y_test.cpu()
plot_predictions(X_train, X_test, y_train, y_test, y_test_preds)

plt.plot(epoch_count, np.array(torch.tensor(train_loss_value).numpy()), label="Train loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_value).numpy()), label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
