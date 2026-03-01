import numpy as np
from sklearn.preprocessing import StandardScaler
from train import run_experiment

# mock data (replace this with your actual dataset from week1)
X = np.random.rand(1000, 10)
y = np.random.rand(1000, 1)

# scale your data for the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# experiment 1: shallow network with ReLU
config_1 = {
    "hidden_layers": [32],
    "activation": "ReLU",
    "learning_rate": 0.01,
    "epochs": 50
}

# experiment 2: deep network with Tanh
config_2 = {
    "hidden_layers" : [64,32,16],
    "activation": "Tanh",
    "learning_rate": 0.001,
    "epochs":100
}

if __name__ == "__main__":
    print("Starting experiment 1...")
    run_experiment(X_scaled, y, config_1)
    
    print("Starting experiment 2...")
    run_experiment(X_scaled, y, config_2)