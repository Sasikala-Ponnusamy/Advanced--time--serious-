# Advanced--time--serious-
Advanced Time series Forecasting using LSTM Neural Network

This repository contains a compact example of a multivariate time-series forecasting pipeline implemented in a single Jupyter notebook:

Time_Series_Forecasting.ipynb — data generation, preprocessing, LSTM model training, evaluation, and explainability using Captum.
Open the notebook in Colab: https://colab.research.google.com/drive/1Cb6gyQcImNLxQRbtU1I2EyTtAoF3m9Mb#scrollTo=6J6-fi7Fcgpm
Repository structure
Time_Series_Forecasting.ipynb — full runnable example (data generation -> preprocess -> dataset -> LSTM training -> evaluation -> Integrated Gradients explainability)
README.md — this file
Summary
The notebook demonstrates generating synthetic multivariate time series data with trend, weekly/yearly seasonality, noise, and a structural break. It trains an LSTM-based forecasting model to predict a 30-day horizon of the first feature using a rolling input window (60 timesteps). The notebook also demonstrates model explainability using Integrated Gradients from Captum.

Key components (what the code does)
Data generation

Generates 3 features:
feature_1: trend + yearly seasonality
feature_2: weekly seasonality + noise
feature_3: structural break + noise
Default length: 1095 days (3 years)
Preprocessing

Standard scaling (StandardScaler)
Sliding-window creation:
window_size = 60 (input timesteps)
forecast_horizon = 30 (output timesteps for feature_1)
Result: X shape (N_windows, 60, 3) and y shape (N_windows, 30)
Dataset & DataLoader

PyTorch Dataset wrapper (TimeSeriesDataset) exposes (X, y) pairs
DataLoader used for batching and shuffling during training
Model

LSTMForecast: single-layer LSTM -> final hidden state -> Linear to forecast horizon
Example hyperparameters in the notebook:
input_size = 3
hidden_size = 64
output_size = 30
batch_size = 32
epochs = 20
optimizer = Adam (lr=0.001)
loss = MSELoss
Training loop implemented in model.train_model
Evaluation

RMSE and MAE computed on the test set
Explainability

Captum IntegratedGradients to compute feature attributions for an input sample
Underlying model (brief)
LSTM (Long Short-Term Memory) recurrent network processes the input sequence of multivariate features. The notebook uses the final hidden state of the LSTM (after processing the 60-step input window) and passes it through a linear layer to produce the 30-step forecast for the target feature.
This approach treats forecasting as sequence-to-vector (predicting a fixed-length forecast horizon per window). It is simple, effective for many problems, and easy to extend (stacked LSTMs, seq2seq with attention, transformers, etc.).
Packages & brief explanation
numpy — numerical operations, array handling
pandas — tabular time-series handling and convenience
torch (PyTorch) — model, training, data utilities
scikit-learn — StandardScaler (feature scaling) and metrics (MSE/MAE)
captum — model explainability for PyTorch models (Integrated Gradients)
(Optional) matplotlib / seaborn — plotting results (not required but useful for visualization)
Recommended install (example):

Create a venv / conda env and install:
pip install numpy pandas scikit-learn torch captum
(If you want reproducible envs, add a requirements.txt or environment.yml file with pinned versions.)

How to run
Open the notebook in Colab:
https://colab.research.google.com/github/NithyaprasathS/time_series_forecasting/blob/main/Time_Series_Forecasting.ipynb
Or run locally:
Create a new virtual environment
pip install numpy pandas scikit-learn torch captum
Open the notebook in Jupyter and run the cells
The notebook includes a full end-to-end runnable example; adjust hyperparameters at the bottom (model, training, window_size, forecast_horizon) if desired.
Example minimal pip command:

pip install numpy pandas scikit-learn torch captum
Hyperparameters (used in the notebook)
window_size = 60
forecast_horizon = 30
hidden_size = 64
batch_size = 32
epochs = 20
learning_rate = 0.001
Simple pipeline diagram (flow)
Data generation -> Preprocessing (scaler, windows) -> Dataset -> DataLoader -> LSTM model -> Forecast output -> Evaluate (RMSE, MAE)
-> Explainability (Captum Integrated Gradients)

ASCII flow: [Generate synthetic data] | [StandardScaler & rolling windows] | [PyTorch Dataset] -> [DataLoader] | [LSTM model] --> [Linear layer] --> [30-step forecast] | [Evaluation: RMSE / MAE] | [Explainability: Captum IG]

Notes & next steps / suggestions
Add a requirements.txt with fixed package versions for reproducibility.
Consider supporting different forecasting modes (seq2seq, teacher forcing, multi-step decoder) or adding early stopping and model checkpointing.
Add visualization cells to show predicted vs actual time series and attribution heatmaps.
License
Pick a license (e.g., MIT) and add LICENSE if you plan to publish the code.
