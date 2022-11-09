
from torch import nn
import torch

class GRU(nn.Module):
    
    def __init__(self):

        super(GRU, self).__init__()
        input_dim  = 1
        hidden_dim = 32
        num_layers = 1
        output_dim = 2
        pass

        layer = dict()        
        layer['gru-gen-01'] = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
            bias=True, batch_first=True, bidirectional=False
        )
        layer['gru-con-01'] = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
            bias=True, batch_first=True, bidirectional=False
        )
        # layer['gru-low-01'] = nn.GRU(
        #     input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
        #     bias=True, batch_first=True, bidirectional=False
        # )
        # layer['gru-close-01'] = nn.GRU(
        #     input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
        #     bias=True, batch_first=True, bidirectional=False
        # )                        
        layer['fc-gen-02'] = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        layer['fc-con-02'] = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        # layer['fc-low-02'] = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        # layer['fc-close-02'] = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        layer['fc-out'] = nn.Sequential(nn.Linear(2*hidden_dim, output_dim), nn.Tanh())
        pass
    
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        layer = self.layer
        train_set_x = x
        x_gen = train_set_x['generation'].unsqueeze(-1)
        x_con = train_set_x['consumption'].unsqueeze(-1)

        h_gen = torch.zeros((1, x_gen.size(0), 32))
        h_con = torch.zeros((1, x_con.size(0), 32))

        o_gen, _ = layer['gru-gen-01'](x_gen, h_gen)
        o_con, _ = layer['gru-con-01'](x_con, h_con)

        o2_gen = layer['fc-gen-02'](o_gen[:,-1,:])
        o2_con = layer['fc-con-02'](o_con[:,-1,:])

        o3 = torch.cat([o2_gen, o2_con], axis=1)
        out = layer['fc-out'](o3)
        print(out.shape)
        return out

    pass