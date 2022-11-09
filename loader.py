
# import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class data:

    def __init__(self, train, test):

        self.train = pd.read_csv(train, header=None)
        self.test = pd.read_csv(test, header=None)
        pass
    
    def scaler(self):

        self.scaler_fun = {
            1: MinMaxScaler(feature_range=(-1, 1)), # Gen
            2: MinMaxScaler(feature_range=(-1, 1)), # Con
            # 2: MinMaxScaler(feature_range=(-1, 1)), # Low
            # 3: MinMaxScaler(feature_range=(-1, 1))  # Close
        }
        tag = [1, 2]
        for i in tag:

            self.train[[i]] = self.scaler_fun[i].fit_transform(self.train[[i]])
            continue

        for i in tag:

            self.test[[i]] = self.scaler_fun[i].transform(self.test[[i]])
            continue

    def initialize(self, back=19, infer=1, valid_size=0.2):
        '''
        back: Track the last 19 days, default 19.
        infer: Infer the one day, default 1.
        valid_size: Validation size, default 0.2.
        '''
        # train = pd.read_csv("training_data.csv", header=None)
        # test  = pd.read_csv("testing_data.csv", header=None)
        train, test = self.train, self.test
        status = pd.DataFrame({4:(len(train) * ['train']) + (len(test) * ['test'])})
        combination = pd.concat([train, test]).reset_index(drop=True)
        combination = pd.concat([combination, status], axis=1)
        #        0           1           2           3         status
        # 0      63.750000   63.849998   63.049999   63.150002  train
        # 1      63.650002   64.000000   63.400002   63.900002  train
        # 2      64.500000   65.150002   64.500000   65.000000  train
        # 3      65.000000   65.000000   64.550003   64.849998  train
        # 4      64.050003   64.099998   63.200001   63.450001  train
        # ...          ...         ...         ...         ...    ...
        # 1239  105.449997  106.349998  105.300003  106.199997   test
        # 1240  106.699997  107.699997  106.699997  107.050003   test
        # 1241  107.349998  107.599998  106.199997  107.099998   test
        # 1242  107.050003  107.199997  106.449997  106.699997   test
        # 1243  106.500000  106.500000  105.099998  105.699997   test        
        lookback = back + infer
        sequence = []
        loop = range(len(combination) - (lookback-1))
        for index in loop: 
            
            sequence.append(combination[index: index + lookback])
            continue

        # tag = ['open', "high", 'low', 'close']
        train_x = {'time':[], 'generation':[], 'consumption':[]}
        train_y = {'time':[], 'generation':[], 'consumption':[]}
        test_x  = {'time':[], 'generation':[], 'consumption':[]}
        test_y  = {'time':[], 'generation':[], 'consumption':[]}
        for item in sequence:
            
            history = item.iloc[:back,:]
            future = item.iloc[-infer:,:]
            history_t, history_g, history_c, history_s = history.transpose().values.tolist()
            future_t, future_g, future_c, future_s = future.transpose().values.tolist()
            if((history_s[-1] == 'train') and (future_s[-1]=='train')):

                train_x['time']        += [history_t]
                train_x['generation']  += [history_g]
                train_x['consumption'] += [history_c]
                pass

                train_y['time']        += [future_t]
                train_y['generation']  += [future_g]
                train_y['consumption'] += [future_c]
                pass

            elif((history_s[-1] == 'test') and (future_s[-1]=='test')):

                test_x['time']        += [history_t]
                test_x['generation']  += [history_g]
                test_x['consumption'] += [history_c]
                pass

                test_y['time']        += [future_t]
                test_y['generation']  += [future_g]
                test_y['consumption'] += [future_c]
                pass
            
            continue
        
        tag = ['time', 'generation', 'consumption']
        for k in tag:
    
            train_x[k] = np.array(train_x[k])
            train_y[k] = np.array(train_y[k])
            test_x[k]= np.array(test_x[k])
            test_y[k]= np.array(test_y[k])
            # train_x['open'].shape
            # (1205, 19)
            # train_y['open'].shape
            # (1205, 1)
            # test_x['open'].shape
            # (19, 19)
            # test_y['open'].shape
            # (19, 1)
            continue

        ##  Split the validation.
        if(valid_size):
            
            train_total = len(train_x['time'])
            valid_set_size = int(np.round(valid_size*train_total))
            train_set_size = train_total - valid_set_size
            train_set_x, train_set_y = {}, {}
            valid_set_x, valid_set_y = {}, {}
            for k in tag:

                if(k=="time"):continue
                train_set_x[k] = torch.from_numpy(train_x[k][:train_set_size,:]).type(torch.Tensor)
                train_set_y[k] = torch.from_numpy(train_y[k][:train_set_size,:]).type(torch.Tensor)
                valid_set_x[k] = torch.from_numpy(train_x[k][train_set_size:,:]).type(torch.Tensor)
                valid_set_y[k] = torch.from_numpy(train_y[k][train_set_size:,:]).type(torch.Tensor)
                test_x[k] = torch.from_numpy(test_x[k]).type(torch.Tensor)
                test_y[k] = torch.from_numpy(test_y[k]).type(torch.Tensor)            
                continue

            output = (train_set_x, train_set_y, valid_set_x, valid_set_y, test_x, test_y)

        else:

            for k in tag:

                if(k=="time"):continue
                train_x[k] = torch.from_numpy(train_x[k]).type(torch.Tensor)
                train_y[k] = torch.from_numpy(train_y[k]).type(torch.Tensor)
                test_x[k] = torch.from_numpy(test_x[k]).type(torch.Tensor)
                test_y[k] = torch.from_numpy(test_y[k]).type(torch.Tensor)
                continue
            
            output = (train_x, train_y, test_x, test_y)

        return(output)

    pass