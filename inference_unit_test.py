

import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import loader
import model

weight_file = 'checkpoint/weight.pt'
loader_data = loader.data('training_data.csv', 'testing_data.csv')
loader_data.scaler()

train_x, train_y, test_x, test_y = loader_data.initialize(back=19, infer=1, valid_size=0)
del train_x, train_y, test_y

model_gru = model.GRU()
model_gru.load_state_dict(torch.load(weight_file))
model_gru.eval()
day = 19
day_open_price_prediction = []
for i in range(day):

    test_input = {
        'open':test_x['open'][i:i+1], 
        'high':test_x['open'][i:i+1],
        'low':test_x['low'][i:i+1], 
        'close':test_x['close'][i:i+1]        
    }
    test_score = model_gru(test_input).detach().numpy()
    test_next_price = loader_data.scaler_fun[0].inverse_transform(test_score).flatten().item()
    pass

    day_open_price_prediction += [test_next_price]
    if(i==0): act, stat = [1], [1]
    elif(i!=0 and i<(day-1)):
    
        look_up = (day_open_price_prediction[i-1] - day_open_price_prediction[i] > 0)
        if(look_up):

            if(stat[i-1]==1): 
                
                act += [-1]
                stat += [0]
                pass

            else:

                act += [0]
                stat += [0]
                pass
        else:
            
            if(stat[i-1]==1): 
            
                act += [0]
                stat += [1]
                pass

            else:
            
                act += [1]
                stat += [1]
                pass    

            pass

        pass

    else:

        if(stat[i-1]==1):

            act += [-1]
            stat += [0]
            pass
        
        else:

            act += [0]
            stat += [0]           
            pass

        pass

    continue

print(act)

