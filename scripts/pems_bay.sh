#!/bin/bash
python train.py \
--use_ema  \
--epsilon 1.0 \
--learning_rate 0.0005 \
--start_iter 10000 \
--data_id PEMS-BAY \
--data data/PEMS-BAY \
--num_nodes 325 \
--adjdata data/sensor_graph/adj_mx_bay.pkl \
--device cuda:0 \
--gcn_bool \
--adjtype doubletransition \
--addaptadj  \
--randomadj