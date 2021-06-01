# VB & Calibrated Neural Nets
Repository containing code and experiments for the paper "**Probabilistic Neural Network Versus Ex-Post Calibration For Capturing Prediction Uncertainty**".

![alt text](/vb_example.png)



The core files are as follows:
- `run_experiments`: Entry program. Starts the experiments.
- `config.yaml`: Contains program configurations and training hyperparameters.
- `trainer.py`: VB trainer.
- `net.py`: Standard neural network builder.
- `models.py`: Neural network architecture.
- `dataset.py`: TF dataset builder.
- `utils.py`: General utility functions for plotting, exporting arrays etc.
- `vbutils`: Contains utility functions related to VB.
- `betacal`: Contains calibration methods (Beta, Logistic and Isotonic). Code adapted from [here](https://github.com/betacal/aistats2017/tree/master/experiments).

All results and logs can be access under `results/` and `logs/` folders (both are automatically created).

## Setup
> Note: Python version == 3.6

Main dependencies:
- TensorFlow 2.0
- Scikit-learn
- Numpy

Please refer to `requirements.txt` file for all the dependencies for this project. To setup the program on your local machine:

1. Clone the repo:
```
git clone https://github.com/sodalabsio/vb_calibration_nn.git
cd vb_calibration_nn/
```

2. Install libraries:
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Execute `run_experiments.sh` by providing the path to activate the virtual environment:
```
./run_experiments.sh venv/bin/activate
```

## References
- Datasets were downloaded and preprocessed using the following [code](https://github.com/REFRAME/betacal/blob/master/aistats2017/experiments/data_wrappers/datasets.py) as per previous [work](http://proceedings.mlr.press/v54/kull17a.html).

## License
MIT