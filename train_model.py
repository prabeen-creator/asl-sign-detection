import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from mlp import MLP

def run_experiment(X_train, y_train, config):
    # initialize W&B run
    wandb.init(project="Neural-network-project", config = config)
    curr_config = wandb.config
    
    # initialize model
    model = MLP(
        input_size= X_train.shape[1],
        hidden_layers= curr_config.hidden_layers,
        output_size=1, #change to number of classes if classification
        activation_type= curr_config.activation
    )
    
    criterion = nn.MSELoss() # use nn.CrossEntropyLoss() for classification
    optimizer = optim.Adam(model.parameters(), lr = curr_config.learning rate)
    
    # convert data to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).view(-1,1)
    
    for epoch in range(curr_config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # log to W&B
        wandb.log({"loss":loss.item(), "epoch": epoch})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    wandb.finish()