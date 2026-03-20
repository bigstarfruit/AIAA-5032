# STGCN in PyTorch

PyTorch reimplementation and extension of STGCN for traffic forecasting on PeMSD7.

This repository is now a pure PyTorch codebase with a single active training and evaluation pipeline.

## What is included

- `stgcn`: STGCN model with configurable graph approximation and spatial ablation
- `persistence`: last-value baseline
- `temporal_mlp`: non-graph temporal baseline
- `lstm`: recurrent baseline
- unified experiment runner with per-model output directories
- visualization script for training curves, forecast plots, and error heatmaps
- group project plan with suggested experiments and team split

## Repository layout

```text
.
|-- data_loader/
|-- dataset/
|-- engine/
|-- models/
|   |-- baselines/
|   `-- stgcn/
|-- output/
|   |-- experiments/
|   `-- visualizations/
|-- scripts/
|-- utils/
|-- group_project_plan.md
|-- main.py
|-- requirements.txt
`-- README.md
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main runtime dependencies:

- PyTorch
- NumPy
- SciPy
- Pandas
- Matplotlib
- TensorBoard

## Dataset

Supported layout:

- `dataset/PeMSD7_V_228.csv`
- `dataset/PeMSD7_W_228.csv`
- `dataset/PeMSD7_V_1026.csv`
- `dataset/PeMSD7_W_1026.csv`

or the extracted archive layout:

- `dataset/PeMSD7_Full/PeMSD7_V_228.csv`
- `dataset/PeMSD7_Full/PeMSD7_W_228.csv`
- `dataset/PeMSD7_Full/PeMSD7_V_1026.csv`
- `dataset/PeMSD7_Full/PeMSD7_W_1026.csv`

The default split follows the original setup:

- 34 days train
- 5 days validation
- 5 days test

## Run experiments

### STGCN baseline

```bash
python main.py --model_name stgcn --exp_name baseline --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### Persistence baseline

```bash
python main.py --model_name persistence --exp_name baseline_persistence --n_route 228 --epoch 1 --device cpu
```

### Temporal MLP baseline

```bash
python main.py --model_name temporal_mlp --exp_name baseline_mlp --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

### LSTM baseline

```bash
python main.py --model_name lstm --exp_name baseline_lstm --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

### STGCN with first-order graph approximation

```bash
python main.py --model_name stgcn --exp_name ablation_first --graph_approx first --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### STGCN without spatial graph convolution

```bash
python main.py --model_name stgcn --exp_name ablation_no_spatial --use_spatial false --n_route 228 --epoch 50 --batch_size 16 --device cuda
```

### Direct multi-step baseline example

```bash
python main.py --model_name temporal_mlp --exp_name direct_mlp --direct_multi_step true --n_route 228 --epoch 50 --batch_size 32 --device cuda
```

## Output structure

Each run writes to:

```text
output/experiments/<exp_name>/<model_name>/
```

Typical files:

- `best.pt`
- `latest.pt`
- `epoch_*.pt`
- `train.log`
- `test.log`
- `history.json`
- `test_results.json`
- `run_meta.json`
- `best_meta.json`
- `experiment_manifest.json`
- `tensorboard/`

## Visualization

Generate figures from an experiment directory:

```bash
python scripts/visualize_results.py   --run_meta output/experiments/baseline/stgcn/run_meta.json   --history output/experiments/baseline/stgcn/history.json   --test_results output/experiments/baseline/stgcn/test_results.json   --checkpoint_dir output/experiments/baseline/stgcn   --output_dir output/visualizations
```

The script can generate:

- training curves
- horizon error bars
- rollout error curves
- single-sensor forecast plots
- spatio-temporal error heatmaps
- adjacency matrix heatmap

## Notes on code status

Active code path:

- `main.py`
- `engine/`
- `models/stgcn/`
- `models/baselines/`
- `scripts/visualize_results.py`

This is the supported PyTorch pipeline.

## Group project support

See [group_project_plan.md](./group_project_plan.md) for:

- baseline suggestions
- ablation matrix
- scaling experiments
- 3/4/5-person team split

## Citation

If you use the original method, please cite the IJCAI 2018 paper:

```bibtex
@inproceedings{yu2018spatio,
  title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
  author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
  booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence},
  year={2018}
}
```
