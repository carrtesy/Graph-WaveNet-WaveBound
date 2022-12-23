#!/bin/bash
python train.py \
--use_ema \
--epsilon 1.0 \
--start_iter 5000 \
--learning_rate 0.001 \
--data_id METR-LA \
--data data/METR-LA \
--num_nodes 207 \
--adjdata data/sensor_graph/adj_mx.pkl \
--device cuda:0 \
--gcn_bool \
--adjtype doubletransition \
--addaptadj  \
--randomadj