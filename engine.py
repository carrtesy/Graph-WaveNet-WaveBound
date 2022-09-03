import torch.optim as optim
from model import *
import util
from utils.ema import EMAUpdater
import numpy as np
import torch
import torch.nn as nn

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes,
                 nhid , dropout, lrate, wdecay, device, supports,
                 gcn_bool, addaptadj, aptinit, args):

        self.args = args
        self.iter_count = 0
        self.source_model = gwnet(device, num_nodes, dropout,
                                  supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                                  aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                                  dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.source_model.to(device)

        # ema
        self.use_ema = self.args.use_ema
        if self.use_ema:
            self.target_model = gwnet(device, num_nodes, dropout,
                                      supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                                      aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                                      dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
            self.target_model.to(device)
            self.ema_updater = EMAUpdater(self.target_model, self.source_model,
                                          self.args.moving_average_decay,
                                          self.args.start_iter)
            self.eval_model = self.target_model
            self.eval_model.to(device)
        else:
            self.eval_model = self.source_model
            self.eval_model.to(device)

        self.optimizer = optim.Adam(self.source_model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5


    def train(self, input, real_val):
        if self.args.use_ema:
            self.target_model.train()
        self.source_model.train()

        input = nn.functional.pad(input,(1,0,0,0))

        # get source, target output
        real = torch.unsqueeze(real_val, dim=1)
        source_output = self.source_model(input)
        source_output = source_output.transpose(1, 3)
        source_pred = self.scaler.inverse_transform(source_output)
        if self.use_ema and self.iter_count >= self.args.start_iter:
            target_output = self.target_model(input)
            target_output = target_output.transpose(1, 3)
            target_pred = self.scaler.inverse_transform(target_output)
            loss = self.compute_loss(source_pred, target_pred, real, null_val=0.0)
        else:
            loss = self.loss(source_pred, real, 0.0)

        self.optimizer.zero_grad()
        # compute loss
        loss.backward()

        # optim
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.source_model.parameters(), self.clip)
            #if self.use_ema:
            #    torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), self.clip)
        self.optimizer.step()

        # ema update
        if self.use_ema:
            self.ema_updater.update(iter_count=self.iter_count)

        # training log
        mape = util.masked_mape(source_pred, real, 0.0).item()
        rmse = util.masked_rmse(source_pred, real, 0.0).item()
        return loss.item(), mape, rmse

    def compute_loss(self, source_output, target_output, y, null_val=np.nan):
        '''
        if np.isnan(null_val):
            mask = ~torch.isnan(y)
        else:
            mask = (y != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        '''

        source_loss = self.loss(source_output, y, null_val=null_val, reduce=False).mean(0)
        target_loss = self.loss(target_output, y, null_val=null_val, reduce=False).mean(0)
        loss = torch.abs(source_loss - target_loss + self.args.epsilon).mean()
        return loss

    @torch.no_grad()
    def eval(self, input, real_val):
        self.eval_model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.eval_model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
