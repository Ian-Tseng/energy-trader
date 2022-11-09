#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import torch
import os
# from torch import nn
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
import time
import argparse
import loader ##  loader data and process.
import model  ##  Model architecture.


import csv
import glob
import random



back = 10
infer = 1
skip_min = 5
end_max = 0.2

class Trader:

    def __init__(self, train_path=None, test_path=None, generation_data_path= None, consumption_data_path= None):

        self.train_path = train_path
        self.test_path = test_path
        self.generation_data_path= generation_data_path
        self.consumption_data_path= consumption_data_path

    def load(self):

        loader_data = loader.data(self.test_path, self.test_path)
        loader_data.scaler()
        tr_x, tr_y, te_x, _ = loader_data.initialize(
            back=back, infer=infer, valid_size=0
        )
        self.scale_fun = loader_data.scaler_fun#[0].inverse_transform
        self.train_x = tr_x
        self.train_y = tr_y
        self.test_x = te_x
        return

    def load_model(self, path):

        # model_gru = model.GRU()
        # model_gru.load_state_dict(torch.load(path))
        model_gru = torch.load(path)
        
        model_gru.eval()
        self.model_gru = model_gru
        return

    def train(self):

        train_set_x = self.train_x
        train_set_y = self.train_y
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

            train_y_hat = model_gru(train_set_x)
            train_y = torch.cat([train_set_y['generation'], train_set_y['consumption']], axis=1)
            
            loss = criterion(train_y_hat, train_y)
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

        self.model_gru = model_gru.eval()
        return

    def predict(self, test_input):

        self.model_gru.eval()
        test_score = self.model_gru(test_input).detach().numpy()
        # print(test_score[:,0:1].shape)

        test_gen_prediction = self.scale_fun[1].inverse_transform(test_score[:,0:1])
        test_con_prediction = self.scale_fun[2].inverse_transform(test_score[:,1:2])
        return([test_gen_prediction, test_con_prediction])

    
def init_predict(generation_data_path, consumption_data_path):
  
   # training_dir = os.path.join(os.getcwd(), "energy_data")
    testing_gen = generation_data_path 
    testing_con = consumption_data_path 
    model_path = os.path.join(os.getcwd(), "model.h5")
    predict_data_path = os.path.join(os.getcwd(), "predict_data.csv")
   
        
    train_data_list= []
  
    test_path = os.path.join(os.getcwd(), 'test_data.csv')
   
    test_gen = pd.read_csv(testing_gen, header=None)
    test_gen= test_gen[1:]
    test_gen.columns = ['time', 'generation']
    test_con = pd.read_csv(testing_con, header=None)
    test_con= test_con[1:]
    test_con.columns = ['time', 'sonsumption']
    test_data = pd.merge(test_gen, test_con, how='inner', on='time')
   
    test_data= test_data[:24]
    out = test_data[['time']].copy()
    print ('test_data', test_data[:24])
    
    test_data.to_csv(test_path, header=None, index=False)
    
    trader = Trader(train_path=None, test_path=test_path)
    trader.load()
    trader.load_model(path=model_path)

    max_length = len(trader.test_x['time'])
    scale_fun = trader.scale_fun
    loop = range(max_length)  ##  Test data length.
    
    # scale_fun = trader.scale_fun
    # max_length = len(trader.test_x['open'])
    # endpoint = max_length - int(max_length * end_max)
    # scale_fun = trader.scale_fun
    test_x = trader.test_x
    # skip = min(back, skip_min)
    # act_list = []
    # have = False
    # out = pd.DataFrame()
    time_slot = []
    gen = []
    con = []
    for d in loop:
        # print(test_x['time'])
        # if(d>0): time_slot += [test_x['time'][d:d+1][0][0]]
        test_input = {
            'generation':test_x['generation'][d:d+1],
            'consumption':test_x['consumption'][d:d+1]
        }
        # now_open_price = scale_fun[1].inverse_transform(test_input['generation']).flatten()[-1].item()
        # now_open_price = scale_fun[2].inverse_transform(test_input['consumption']).flatten()[-1].item()
        next_day_pred = trader.predict(test_input)
        # print(next_day_pred)
        # next_day_gen = scale_fun[1].inverse_transform(next_day_pred[0])
        # next_day_con = scale_fun[2].inverse_transform(next_day_pred[1])
        next_day_gen = next_day_pred[0].item()
        next_day_con = next_day_pred[1].item()
        # print(next_day_con.item())
        # next_day_time = test_input['time']
        # print(test_x['time'][d:d+1])
        # print(test_x['time'][d:d+1][0][0])
        # if(d==(max_length-1)): time_slot += [test_x['time'][d:d+1][0][0][1]]
        # row = pd.DataFrame({"time":time_slot, "generation": [next_day_gen.item()], "consumption":[next_day_con.item()]})
        gen += [next_day_gen]
        con += [next_day_con]
        # out = pd.concat([out,row],axis=0)
        continue
        # print(next_day_gen)
        # print(next_day_time)
    # print(out)
    out_gencon = pd.concat([out, pd.DataFrame({"generation": gen, "consumption":con})], axis=1)
    # print(out_gencon)
    out_gencon.to_csv(predict_data_path, index=False)
            
    

   
    
    


# In[28]:


#################################################################################################################################
# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return
# You should not modify this part.
#################################################################################################################################



def gen_val_data(total_data_list, consumption_data_name, generation_data_name):
    a= 300
    b= a+ 50                           
    rand_range_start= random.randrange(a, b)
  
    new_data_list= total_data_list[rand_range_start:]
    data_len= 7* 24
    target_data_list= []
    for i, c in enumerate(new_data_list):
        time_in_h= c[0].split(' ')[1]
       
        if time_in_h== '00:00:00':
            target_data_list= new_data_list[i:i+ data_len]
            break
    generation_data= [[i[0], i[1]] for i in target_data_list]
    consumption_data= [[i[0], i[2]] for i in target_data_list]
    
    generation_data_dir= os.path.join(os.getcwd(), generation_data_name)            
    consumption_data_dir= os.path.join(os.getcwd(), consumption_data_name)
    
    with open(generation_data_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(generation_data)
    with open(consumption_data_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(consumption_data)
    
        
 

def get_all_data(target_dir):
    data_dir_index_list= []
  
    for i in glob.iglob(os.path.join(target_dir, "*csv")):
        title, ext= os.path.splitext(os.path.basename(i))
        if not 'target' in title:
            continue
     
        data_dir_index_list.append(i)
 
    total_data_list= []
    for data_dir in data_dir_index_list:
        with open(data_dir, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data_list= [i for i in reader]
            total_data_list+= data_list[1:]
            
    return total_data_list
           
        
def gen_bidresult(bid_result_data_name):
    bid_result_data_dir= os.path.join(os.getcwd(), bid_result_data_name)
    header= ['time', 'action', 'target_price', 'target_volume', 'trade_price', 'trade_volume', 'status']
    with open(bid_result_data_dir, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        
def get_peak_time(data_list):
    
    # For consumption peak
    data_list= sorted(data_list[1:],  key = lambda s: float(s[2]))
    high_peak_time_list= []
    for i in data_list[0: 1500]:
        time_in_h= i[0].split(' ')[1]
        high_peak_time_list.append(time_in_h)
    
    unique, counts = np.unique(np.array(high_peak_time_list), return_counts=True)
    high_peak_time_count= dict(zip(unique, counts))

    high_peak_time_count= {k: v for k, v in (sorted(high_peak_time_count.items(), key=lambda item: item[1], reverse=True))}
    
    
    
    
    loew_peak_time_data_list= sorted(data_list[1:],  key = lambda s: float(s[2]), reverse=True)  
    low_peak_time_list= []
    for i in loew_peak_time_data_list[0: 1500]:
      
        low_peak_time_list.append(i[0].split(' ')[1])
    unique, counts = np.unique(np.array(low_peak_time_list), return_counts=True)
    low_peak_time_count= dict(zip(unique, counts))
    low_peak_time_count= {k: v for k, v in (sorted(low_peak_time_count.items(), key=lambda item: item[1], reverse=True))}
   
    
    
    # # For generate peak 
    high_generate_peak_time_data_list= sorted(data_list[1:],  key = lambda s: float(s[1]), reverse= True)
    high_peak_time_list= []
    for i in high_generate_peak_time_data_list[0: 1500]:

        time_in_h= i[0].split(' ')[1] 
        high_peak_time_list.append(time_in_h)
    
    unique, counts = np.unique(np.array(high_peak_time_list), return_counts=True)
    high_peak_time_count= dict(zip(unique, counts))

    high_peak_time_count= {k: v for k, v in (sorted(high_peak_time_count.items(), key=lambda item: item[1], reverse=True))}
   
    
    
    
    low_generate_peak_time_data_list= sorted(data_list[1:],  key = lambda s: float(s[1]))
    low_peak_time_list= []
    for i in low_generate_peak_time_data_list[0: 1500]:
 
        low_peak_time_list.append(i[0].split(' ')[1])
    unique, counts = np.unique(np.array(low_peak_time_list), return_counts=True)
    low_peak_time_count= dict(zip(unique, counts))
    low_peak_time_count= {k: v for k, v in (sorted(low_peak_time_count.items(), key=lambda item: item[1], reverse=True))}
   
    
def get_gen_consum_progress(data_list):
    next_day_count=0 
    total_gen= 0
    total_consum= 0
    
    for i, data in enumerate(data_list):
        time= data[0].split(' ')[1]
        if time== '00:00:00' and i!= 0:
            next_day_count+= 1
            break
        total_gen+= float(data[1])
        total_consum+= float(data[2])
        print ('time', time, 'total_gen', total_gen, 'total_consum', total_consum)
        
def get_request_gen_rate(data_list):
    
    # For consumption peak
    data_list= sorted(data_list[1:],  key = lambda s: float(s[2]))
    high_peak_time_list= []
    request_gen_rate_dict= dict()
    total_gen= 'total_gen'
    total_consum= 'total_consum'
   
    for i, c in enumerate(data_list):
        
        request_gen_rate= 'request_gen_rate'
        
        
            
        time_in_h= c[0].split(' ')[1]
        
        if not time_in_h in request_gen_rate_dict:
            request_gen_rate_dict[time_in_h]= {total_gen: 0, total_consum: 0}
            request_gen_rate_dict[time_in_h][total_gen]= float(c[1])
            request_gen_rate_dict[time_in_h][total_consum]= float(c[2])
            
        update_gen= request_gen_rate_dict[time_in_h][total_gen]+ float(c[1])
        update_consum= request_gen_rate_dict[time_in_h][total_consum]+float(c[2])
        request_gen_rate_dict[time_in_h][total_gen]= update_gen
        request_gen_rate_dict[time_in_h][total_consum]= update_consum
       
        if update_gen== 0:
            update_gen= 0.00000001
        update_request_gen_rate= update_consum/ update_gen
        
        request_gen_rate_dict[time_in_h][request_gen_rate]= update_request_gen_rate
            
    
    request_gen_rate_list= []
    for i in request_gen_rate_dict:
        request_gen_rate_list.append([i, request_gen_rate_dict[i][request_gen_rate]])
      
    
    request_gen_rate_list= sorted(request_gen_rate_list,  key = lambda s: float(s[1]))
    for i in request_gen_rate_list:
        print (i)
    return request_gen_rate_list

def check_in_target_time(h, request_gen_rate_list):
    in_buy_time= False
    in_sell_time= False
    sell_range= range(17, 23)
    h_h= int(h[: 2])
   
    request_gen_rate_dict= dict(request_gen_rate_list)
    if request_gen_rate_dict[h]< 1:
        in_buy_time= True
    if request_gen_rate_dict[h]> 200 and h_h in sell_range:
        in_sell_time= True
        
    return in_buy_time, in_sell_time

def get_price(h, action, amount, market_price,request_gen_rate_list):
   
    price_list= []
    bill_list= []
    sell_discount_range= reversed(np.arange(5, 8, 0.5))
    buy_discount_range= np.arange(5, 8, 0.5)
   
    if action== [0, 1]: # Buy
        index_list= [i[0] for i in request_gen_rate_list]
        priority= index_list.index(h)
        for i in buy_discount_range:
            price_list.append(market_price* i/ 10)
            bill_list.append(market_price- (market_price* i/ 10))
    else:   # Sell
        index_list= [i[0] for i in reversed(request_gen_rate_list)]
        priority= index_list.index(h)
        for i in sell_discount_range:
            price_list.append(market_price* i/ 10)
            bill_list.append(market_price* i/ 10)
            

    return price_list, bill_list, priority
    
    
    
        

def init_agent(info_in_hour_list, request_gen_rate_list):
    market_price=   2.4738 # 2.5256
    action= [0, 1] # [1 if Sell 0 else, 1 if Buy 0 else] # default state= [0, 0] or [1, 0]
    # Default upper price= market price
    total_energy= 0
    total_consumption= 0
    ori_energy= 0
    ori_consumption= 0
    action_list= []
    action_history_list= []
    state_history_list= []
    total_bill_arr= np.zeros(6)
    total_bill_buy_arr= np.zeros(6)
    total_bill_sell_arr= np.zeros(6)
    for c, (time , generate, consumption) in enumerate(info_in_hour_list):
        if c== 0 or len(generate)== 0:
            continue
      
        generate= float(generate)
        consumption= float(consumption)
        time_in_h= time.split(' ')[1]
            
        if generate- consumption> 0:
            state_in_h= [1, 0]
        else:
            state_in_h= [0, 0]
        
        
        if len(state_history_list)> 0:
            last_state= state_history_list[-1][1]
            update_state= (np.array(state_in_h)+ np.array(last_state)).tolist()
            for i, c in enumerate(update_state):
                if c>0:
                    update_state[i]= 1
                else:
                    update_state[i]= 0
                    
        else:
            update_state= state_in_h
        
        
        
        total_energy+= generate
        total_consumption+= consumption
        in_buy_time, in_sell_time= check_in_target_time(time_in_h, request_gen_rate_list)
        price= None
        amount= None
        action= None
        action_name= None
        prioirty= None
        
        
        ori_energy+= generate
        ori_consumption+= consumption
        
        amount= total_energy- total_consumption
        if 1 in update_state and total_energy- total_consumption> 0 and in_sell_time and amount!= 0: # Sell action
          
            action= [1, 0]
            total_energy-= amount
            remaining= total_energy- total_consumption
            price_range, bill_list, prioirty= get_price(time_in_h, action, amount, market_price, request_gen_rate_list)
            
            if action== [1, 0]:
                action_name= 'sell'
            else:
                action_name= 'buy'
            for i, price in enumerate(price_range):
                action_list.append([time_in_h, action_name, price, amount])
                action_history_list.append([time_in_h, action, price, amount, remaining, bill_list[i]])
                state_history_list.append([time_in_h, update_state, action, price, amount, remaining, bill_list[i]])
            total_bill_sell_arr= total_bill_sell_arr+ np.array(bill_list)* amount
            
        if update_state== [0, 0] and in_buy_time:  # Buy action
            action= [0, 1]
            amount= abs(amount)
            total_energy+= amount
            remaining= total_energy- total_consumption
            price_range, bill_list, prioirty= get_price(time_in_h, action, amount, market_price, request_gen_rate_list)
            
            if action== [1, 0]:
                action_name= 'sell'
            else:
                action_name= 'buy'
            for i, price in enumerate(price_range):
                action_list.append([time_in_h, action_name, price, amount])
                action_history_list.append([time_in_h, action, price, amount, remaining, bill_list[i]])
                state_history_list.append([time_in_h, update_state, action, price, amount, remaining, bill_list[i]])
            total_bill_buy_arr= total_bill_buy_arr+ np.array(bill_list)* amount
            
        if in_sell_time:  # Append sell
            action= [1, 0]
            amount= abs(total_energy- total_consumption)* 0.8
            if amount== 0:
                continue
            total_energy-= amount
            remaining= total_energy- total_consumption
            price_range, bill_list, prioirty= get_price(time_in_h, action, amount, market_price, request_gen_rate_list)
           
            
            if action== [1, 0]:
                action_name= 'sell'
            else:
                action_name= 'buy'
            for i, price in enumerate(price_range):
                action_list.append([time_in_h, action_name, price, amount])
                action_history_list.append([time_in_h, action, price, amount, remaining, bill_list[i]])
                state_history_list.append([time_in_h, update_state, action, price, amount, remaining, bill_list[i]])
            total_bill_sell_arr= total_bill_sell_arr+ np.array(bill_list)* amount
            
        if in_buy_time:  # Append buy
          
            action= [0, 1]
            amount= abs(total_energy- total_consumption)* 0.8
            total_energy+= amount
            remaining= total_energy- total_consumption
            price_range, bill_list, prioirty= get_price(time_in_h, action, amount, market_price, request_gen_rate_list)
            
            
            if action== [1, 0]:
                action_name= 'sell'
            else:
                action_name= 'buy'
            for i, price in enumerate(price_range):
                action_list.append([time_in_h, action_name, price, amount])
                action_history_list.append([time_in_h, action, price, amount, remaining, bill_list[i]])
                state_history_list.append([time_in_h, update_state, action, price, amount, remaining, bill_list[i]])
            total_bill_buy_arr= total_bill_buy_arr+ np.array(bill_list)* amount
        
        
        print (time, 'action_name', action_name, 'state', update_state, 'price', price, 'amount', amount, 'prioirty', prioirty, 'total_consumption', total_consumption, 'remaining', total_energy- total_consumption, 'ori_remaining', ori_energy- ori_consumption )
        
    for i in action_list:
        print (i)
    total_bill_arr= total_bill_sell_arr- total_bill_buy_arr
   
    print ('total_bill_arr', total_bill_arr)
    return action_list
                                  
                                  
def get_info_in_hour_list(data_list):
    a= 100
    b= a+ 50                           
    rand_range_start= random.randrange(a, b)
    target_validation_data_list= []
    start_add= False
    for i, c in enumerate(data_list[rand_range_start:]):
                    
        time_in_h= c[0].split(' ')[1]
       
        if '00:00:00' == time_in_h:
           
            target_validation_data_list= data_list[rand_range_start+ i: rand_range_start+ i+ 24]
            break
    
    return target_validation_data_list

def get_predicted_data_list(predicted_data_list_dir):
    with open(predicted_data_list_dir, 'r', newline='') as csvfile:
        reader= csv.reader(csvfile, delimiter=',')
        data_list= [i for i in reader]
     #   data_list= [[i[0], float(i[1]), float(i[2])] for i in reader[1:]]

    
    return data_list
    
            
                                  
        
def save_output_data(output_data_dir, data_list):
    header= ['time', 'action', 'target_price', 'target_volume']
    with open(output_data_dir, 'w', newline='') as csvfile:
        writer= csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data_list)


        

  



if __name__ == "__main__":
    args = config()
    
    consumption_data_dir= args.consumption
    generation_data_dir= args.generation
    bidresult_data_dir= args.bidresult
    output_dir= args.output
    predicted_data_dir= os.path.join(os.getcwd(), 'predict_data.csv')
    
    target_dir= os.path.join(os.getcwd(), 'energy_data')
    data_list= get_all_data(target_dir)
    request_gen_rate_list= get_request_gen_rate(data_list) 
    
    init_predict(generation_data_dir, consumption_data_dir)                       
    predict_data_list= get_predicted_data_list(predicted_data_dir)
    action_list= init_agent(predict_data_list, request_gen_rate_list)
    output(output_dir, action_list)
    
    
    




# In[ ]:




