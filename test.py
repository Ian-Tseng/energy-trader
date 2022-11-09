
# from email import header
# from operator import index
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

back = 10
infer = 1
skip_min = 5
end_max = 0.2

class Trader:

    def __init__(self, train_path=None, test_path=None):

        self.train_path = train_path
        self.test_path = test_path

    def load(self):

        loader_data = loader.data(self.train_path, self.test_path)
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

        test_gen_prediction = self.scale_fun[1].transform(test_score[:,0:1])
        test_con_prediction = self.scale_fun[2].transform(test_score[:,1:2])
        return([test_gen_prediction, test_con_prediction])

    def init_test():
        # You should not modify this part.
    

     

        training_dir = os.path.join(os.getcwd(), "inputs/energy_data/")
        testing_gen = os.path.join(os.getcwd(), "inputs/generation/")
        testing_con = os.path.join(os.getcwd(), "inputs/consumption/")
        model_path = os.path.join(os.getcwd(), "models/model.h5")
        output_path = os.path.join(os.getcwd(), "outputs/output.csv")


        pass
        k = 1
        train = pd.concat([pd.read_csv(os.path.join(training_dir, i)) for i in os.listdir(training_dir)][-k:])
        train_path = 'inputs/train_data.csv'
        train.to_csv(train_path, header=None, index=False)
        test_path = 'inputs/test_data.csv'
        # test_time = pd.read_csv(testing_gen, header=None)
        test_gen = pd.read_csv(testing_gen, header=None)
        test_gen.columns = ['time', 'generation']
        test_con = pd.read_csv(testing_con, header=None)
        test_con.columns = ['time', 'sonsumption']
        est_data = pd.merge(test_gen, test_con, how='inner', on='time')
        out = testing_gen[['time']].copy()
    
        testing_gen.to_csv(test_path, header=None, index=False)
    
        trader = Trader(train_path=train_path, test_path=test_path)
        trader.load()
        trader.load_model(path=model_path)
        pass

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
        
            test_input = {
                'generation':test_x['generation'][d:d+1],
                'consumption':test_x['consumption'][d:d+1]
            }
          
            next_day_pred = trader.predict(test_input)
            next_day_gen = scale_fun[1].inverse_transform(next_day_pred[0])
            next_day_con = scale_fun[2].inverse_transform(next_day_pred[1])
       
            gen += [next_day_gen.item()]
            con += [next_day_con.item()]
        
            continue
            # print(next_day_gen)
            # print(next_day_time)
        print(out)
        out_gencon = pd.concat([out, pd.DataFrame({"generation": gen, "consumption":con})], axis=1)
        print(out_gencon)
        out_gencon.to_csv(output_path, index=False)

    if __name__ == '__main__':
        init_test()