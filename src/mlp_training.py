import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import wandb

# 1. Initialize W&B Experiment
# Change 'hidden_size' or 'num_layers' here to see different results on your dashboard
wandb.init(project="asl-sign-detection", config={
    "learning_rate": 0.001,
    "epochs": 20,          # Increased epochs for better learning
    "batch_size": 64,
    "hidden_size": 512,    # Increased from 128 to 512 for Experiment 2
    "dropout_rate": 0.3    # Added dropout to prevent overfitting
})

# 2. Load and Prepare MNIST Data (the link your friend mentioned)
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

def prepare_tensors(df):
    # Normalize pixels (0-255) to (0-1) for faster convergence
    x = torch.tensor(df.drop('label', axis=1).values / 255.0, dtype=torch.float32)
    y = torch.tensor(df['label'].values, dtype=torch.long)
    return x, y

x_train, y_train = prepare_tensors(train_df)
x_test, y_test = prepare_tensors(test_df)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=wandb.config.batch_size, shuffle=True)

# 3. Define a More Complex Architecture (The Experiment)
class ASL_NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ASL_NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(wandb.config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = ASL_NeuralNet(784, wandb.config.hidden_size, 26)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# 4. Training Loop
print(f"Starting Experiment 2: Hidden Size {wandb.config.hidden_size}")
for epoch in range(wandb.config.epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        test_preds = model(x_test)
        _, predicted = torch.max(test_preds, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    
    # Log to W&B Dashboard (Week 2 Requirement)
    wandb.log({
        "epoch": epoch, 
        "loss": train_loss/len(train_loader), 
        "accuracy": accuracy * 100
    })
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Accuracy = {accuracy*100:.2f}%")

# 5. Save the final experiment model
torch.save(model.state_dict(), "asl_mlp_model_v2.pth")
print("Model saved as asl_mlp_model_v2.pth")
wandb.finish()