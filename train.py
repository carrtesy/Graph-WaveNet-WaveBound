import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_id',type=str,default='METR-LA',help='data path')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
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
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')


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

print(args.exp_id)

# checkpoints
if args.use_ema:
    os.makedirs(os.path.join("checkpoints", args.data_id, "wavebound", "source", args.exp_id, "epoch"), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.data_id, "wavebound", "source", args.exp_id, "iter"), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.data_id, "wavebound", "target", args.exp_id, "epoch"), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.data_id, "wavebound", "target", args.exp_id, "iter"), exist_ok=True)
    args.checkpoint_dir = os.path.join("checkpoints", args.data_id, "wavebound")
else:
    os.makedirs(os.path.join("checkpoints", args.data_id, "origin", args.exp_id, "epoch"), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.data_id, "origin", args.exp_id, "iter"), exist_ok=True)
    args.checkpoint_dir = os.path.join("checkpoints", args.data_id, "origin")

device = torch.device(args.device)
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
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

engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, args)

print("start training...",flush=True)
his_loss =[]
val_time = []
train_time = []

best_val_loss = np.inf
best_val_epoch = 0
for i in range(1,args.epochs+1):
    # Training
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx= trainx.transpose(1, 3)
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        metrics = engine.train(trainx, trainy[:,0,:,:])
        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        if iter % args.print_every == 0 :
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        engine.iter_count += 1

    t2 = time.time()
    train_time.append(t2-t1)


    # Validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])


    s2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    val_time.append(s2-s1)
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)

    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

    # save models
    if mvalid_loss < best_val_loss:
        print(f"saving best model @ epoch {i}...")
        if args.use_ema:
            torch.save(engine.source_model.state_dict(), os.path.join(args.checkpoint_dir, "source", args.exp_id, "epoch",  f"{i}.pth"))
            torch.save(engine.source_model.state_dict(), os.path.join(args.checkpoint_dir, "source", args.exp_id, "epoch",  f"best.pth"))
            torch.save(engine.target_model.state_dict(), os.path.join(args.checkpoint_dir, "target", args.exp_id, "epoch", f"{i}.pth"))
            torch.save(engine.target_model.state_dict(), os.path.join(args.checkpoint_dir, "target", args.exp_id, "epoch", f"best.pth"))
        else:
            torch.save(engine.source_model.state_dict(),
                       os.path.join(args.checkpoint_dir, args.exp_id, "epoch", f"{i}.pth"))
            torch.save(engine.source_model.state_dict(),
                       os.path.join(args.checkpoint_dir, args.exp_id, "epoch", f"best.pth"))

        best_val_loss = mvalid_loss
        best_val_epoch = i
    else:
        if args.use_ema:
            torch.save(engine.source_model.state_dict(), os.path.join(args.checkpoint_dir, "source", args.exp_id, "epoch",  f"{i}.pth"))
            torch.save(engine.target_model.state_dict(), os.path.join(args.checkpoint_dir, "target", args.exp_id, "epoch", f"{i}.pth"))
        else:
            torch.save(engine.source_model.state_dict(),
                       os.path.join(args.checkpoint_dir, args.exp_id, "epoch", f"{i}.pth"))

print("Training finished")
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
print(f"The valid loss on best model is {best_val_loss} @ epoch {best_val_epoch}")


print("=" * 50)

# testing
print("Testing")
eval_path = os.path.join(args.checkpoint_dir, args.exp_id, "epoch", "best.pth") if not args.use_ema \
    else os.path.join(args.checkpoint_dir, "target", args.exp_id, "epoch", "best.pth")
engine.eval_model.load_state_dict(torch.load(eval_path))

outputs = []
realy = torch.Tensor(dataloader['y_test']).to(device)
realy = realy.transpose(1, 3)[:, 0, :, :]

for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    with torch.no_grad():
        preds = engine.eval_model(testx).transpose(1, 3)
    outputs.append(preds.squeeze())

yhat = torch.cat(outputs, dim=0)
yhat = yhat[:realy.size(0), ...]

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