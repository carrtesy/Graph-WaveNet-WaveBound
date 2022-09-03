import torch
import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from engine import trainer


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--data_id',type=str,default='METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')

# EMA
parser.add_argument('--use_ema', action='store_true')
parser.add_argument('--epsilon', type=float, default=0.001)
parser.add_argument('--moving_average_decay', type=float, default=0.99)
parser.add_argument('--standing_steps', type=int, default=100)
parser.add_argument('--start_iter', type=int, default=300)
parser.add_argument('--ema_loss', type=str, default='BDFMSE', choices=['DFMSE', 'BDFMSE', 'TDFMSE', 'PDFMSE', 'CSMSE'])
parser.add_argument('--ema_eval_model', type=str, default='target', choices=['source', 'target'])

args = parser.parse_args()

# exp id
args.exp_id = "id_"
args.exp_id += "data_" + str(args.data_id) + "_"
args.exp_id += "ema_" + str(args.use_ema) + "_"
args.exp_id += "eps_" + str(args.epsilon) + "_"
args.exp_id += "mad_" + str(args.moving_average_decay) + "_"
args.exp_id += "sit_" + str(args.start_iter) + "_"
args.exp_id += "lr_" + str(args.learning_rate) + "_"
#args.exp_id += "lr_" + str(args.learning_rate)

print(args.exp_id)

# checkpoints, outputs
if args.use_ema:
    os.makedirs(os.path.join("outputs", args.data_id, "wavebound", args.exp_id, "epoch"), exist_ok=True)
    args.output_dir = os.path.join("outputs", args.data_id, "wavebound")
    args.checkpoint_dir = os.path.join("checkpoints", args.data_id, "wavebound")
else:
    os.makedirs(os.path.join("outputs", args.data_id, "origin", args.exp_id, "epoch"), exist_ok=True)
    args.output_dir = os.path.join("outputs", args.data_id, "origin")
    args.checkpoint_dir = os.path.join("checkpoints", args.data_id, "origin")


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # testing
    print("Testing")
    eval_path = os.path.join(args.checkpoint_dir, args.exp_id, "epoch", "best.pth") if not args.use_ema \
        else os.path.join(args.checkpoint_dir, "target", args.exp_id, "epoch", "best.pth")
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, args)

    engine.eval_model.load_state_dict(torch.load(eval_path))
    engine.eval_model.eval()

    print('model load successfully')

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.eval_model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]




    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    # save output as np
    print(f"save output as numpy... {os.path.join(args.output_dir, args.exp_id, 'epoch')}")

    outp = scaler.inverse_transform(yhat)
    outp = outp.cpu().numpy()
    realy = realy.cpu().numpy()
    with open(os.path.join(args.output_dir, args.exp_id, 'epoch', 'true.npy'), 'wb') as f:
        np.save(f, realy)
    with open(os.path.join(args.output_dir, args.exp_id, 'epoch', 'pred.npy'), 'wb') as f:
        np.save(f, outp)
    print("done.")

    '''
    Heatmaps
        if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb"+ '.pdf')

    y12 = realy[:,99,11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:,99,11]).cpu().detach().numpy()

    y3 = realy[:,99,2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:,99,2]).cpu().detach().numpy()

    df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    df2.to_csv('./wave.csv',index=False)

    '''


if __name__ == "__main__":
    main()
