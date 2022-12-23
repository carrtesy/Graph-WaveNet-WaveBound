# WaveBound + GraphWaveNet

Implementation of [WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting (NeurIPS 2022)](https://arxiv.org/abs/2210.14303), applied to [Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019](https://arxiv.org/abs/1906.00121). This repo is officially provided as examplary code for applying WaveBound method to Traffic domain. To view WaveBound's official implementation, visit the [link-to-wavebound-github](https://github.com/choyi0521/WaveBound).

## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Process raw data 
```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands
```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```
