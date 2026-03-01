import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_type="ReLU"):
        super(MLP, self).__init__()
        
        # select activation function
        if activation_type == "ReLU":
            self.activation = nn.ReLu()
        elif activation_type == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.leakyReLU()
        
        layers = []
        in_dim = input_size
        
        # dynamically build hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_type())
            in_dim = h_dim
         
        # final output layer   
        layers.append(nn.layers(in_dim, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.network(x)
    
    