
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import plotly.graph_objects as go
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler
import math, time
from sklearn.metrics import mean_squared_error

import loader, model

loader_data = loader.data(train = 'training_data.csv', test = 'testing_data.csv')
loader_data.scaler()
back = 10
infer = 1
train_set_x, train_set_y, test_x, test_y = loader_data.initialize(back=back, infer=infer, valid_size=0.0)
_, _ = test_x, test_y  ## Skip test when experiment.
pass

model_gru = model.GRU()
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model_gru.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)
pass

num_epochs = 100 
hist = np.zeros(num_epochs)
start_time = time.time()
model_gru.train()
for t in range(num_epochs):

    train_y_hat_open = model_gru(train_set_x)
    train_y_open = train_set_y['open']
    loss = criterion(train_y_hat_open, train_y_open)
    if((t+1)%10==0): print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    scheduler.step()
    continue

training_time = time.time()-start_time    
print("Training time: {}".format(training_time))
pass

##
model_gru.eval()
os.makedirs("checkpoint", exist_ok=True)
torch.save(model_gru.state_dict(), 'checkpoint/weight.pt')
