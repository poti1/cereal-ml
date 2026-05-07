#!/usr/bin/env python

# Main
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
from pathlib import Path
from copy import deepcopy

# torch.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dl
from torch.utils.data import TensorDataset as td

# sklearn.
from sklearn.compose import make_column_selector as mcs
from sklearn.compose import make_column_transformer as mct
from sklearn.impute import SimpleImputer as si
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import make_pipeline as mp
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import OneHotEncoder as ohe

ON_ANDROID = int(os.getenv('ON_ANDROID', '0'))


class Cereal:
    """Cereal model class"""

    # Setup.
    def __init__(self):
        self.args = self._get_args()
        self.model_file = Path('cereal.pt')
        self.csv_file = Path('cereal.csv')
        self.png_file = Path('loss.png')

        if self.args.train:
            self.model_file.unlink(missing_ok=True)

        self.load()

    def _get_args(self):
        parser = argparse.ArgumentParser(
            description='Cereal ingredients model',
        )

        parser.add_argument(
            '-t',
            '--train',
            action='store_true',
            help='Retrain the model',
        )

        parser.add_argument(
            '-p',
            '--plot',
            action='store_true',
            help='Show loss plot graph',
        )

        parser.add_argument(
            '-d',
            '--data',
            help='Provide custom test data',
        )

        return parser.parse_args()

    def read_csv(self):
        df = pd.read_csv(self.csv_file)

        self.X = df.drop(
            ['calories', 'name', 'mfr', 'type'],
            axis=1,
        )
        self.y = df.calories

    # Split.
    def split_data(self):
        self.make_preprocessing_pipeline()

        # X.
        train_raw, valid, test = self.tts(self.X)
        train = self.fit_transform(train_raw)
        valid = self.transform(valid)
        test = self.transform(test)
        self.X_train_raw = train_raw
        self.X_train = self.to_tensor(train)
        self.X_valid = self.to_tensor(valid)
        self.X_test = self.to_tensor(test)

        # y.
        train, valid, test = self.tts(self.y)
        self.y_train = self.to_tensor(train.values)
        self.y_valid = self.to_tensor(valid.values)
        self.y_test = self.to_tensor(test.values)

    def make_preprocessing_pipeline(self):
        self.pl = mct(
            (self.make_categorical_pipeline(), mcs(dtype_include=object)),
            (self.make_numeric_pipeline(), mcs(dtype_include=np.number)),
        )

    def make_categorical_pipeline(self):
        return mp(
            si(strategy='most_frequent'),
            ohe(handle_unknown='ignore', sparse_output=False),
        )

    def make_numeric_pipeline(self):
        return mp(
            si(strategy='median'),
            ss(),
        )

    def tts(self, data):
        train, temp = tts(data, test_size=0.3, random_state=1)
        valid, test = tts(temp, test_size=0.5, random_state=1)

        return train, valid, test

    def fit_transform(self, data):
        return self.pl.fit_transform(data)

    def transform(self, data):
        return self.pl.transform(data)

    def to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32)

    # Model.
    def make_model(self):

        print('Making model')
        rows, cols = self.X_train.shape

        self.model = nn.Sequential(
            nn.Linear(cols, rows * 8), nn.ReLU(),
            nn.Linear(rows * 8, rows * 4), nn.ReLU(),
            nn.Linear(rows * 4, rows * 2), nn.ReLU(),
            nn.Linear(rows * 2, rows * 1), nn.ReLU(),
            nn.Linear(rows * 1, 1),
        )

    # Train.
    def train_model(self):
        epochs = 10
        patience = 5
        model = self.model

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.L1Loss()  # MAE
        rows = len(self.X_train)

        loader = dl(
            td(self.X_train, self.y_train),
            batch_size=rows // 5,
            shuffle=True,
        )

        best_val = float('inf')
        best_state = deepcopy(model.state_dict())
        wait = 0
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # --- train ---
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= rows

            # --- validate ---
            model.eval()
            with torch.no_grad():
                val_pred = model(self.X_valid).squeeze()
                val_loss = loss_fn(val_pred, self.y_valid).item()

            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch + 1}/{epochs}  loss: {epoch_loss:.4f}  val_loss: {val_loss:.4f}')

            # --- early stopping ---
            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stop at epoch {epoch + 1}')
                    break

        model.load_state_dict(best_state)

        self.history = history

    # Load.
    def load(self):
        model_file = self.model_file

        if model_file.is_file():
            cache = jl.load(model_file)
            self.__dict__.update(
                cache.__dict__,
                args=self.args,
            )
            print(f'Loaded model from: {model_file}')
        else:
            self.read_csv()
            self.split_data()
            self.make_model()
            self.train_model()

            jl.dump(self, model_file)
            print(f'Saved model to: {model_file}')

    # Plot.
    def plot_history(self):
        png_file = self.png_file
        history = self.history

        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(png_file)
        print(f'Saved: {png_file}')

        if ON_ANDROID:
            os.system(f'termux-open {png_file}')
        else:
            plt.show()

    # Predict.
    def predict(self, string):
        X_custom = self.build_custom_data(string)
        self.print_df(X_custom)

        X = self.transform(X_custom)
        X = self.to_tensor(X)
        y = self._predict_df(X).astype(int)

        print(f'\nCalories: {y}')

    def _predict_df(self, X):
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(X).squeeze().numpy()

        return y_pred

    def build_custom_data(self, string):
        X = pd.DataFrame([self.X_train_raw.median()])
        custom = self.str_to_dict(string)
        keys = set(X.columns) & set(custom.keys())
        for k in keys:
            X[k] = custom[k]

        return X

    def print_df(self, X):
        print('Custom Data:')
        print(
            X.T.astype(int).rename(
                {0: ''},
                axis=1,
            ).sort_index()
        )

    def str_to_dict(self, string):
        return dict(
            s.split('=') for s in string.replace(' ', '').split(',')
        )


############################################################
#                     MAIN
############################################################
cereal = Cereal()

if cereal.args.plot:
    cereal.plot_history()

if cereal.args.data:
    cereal.predict(cereal.args.data)
else:
    y_pred = cereal._predict_df(cereal.X_test)
    mae_val = mae(y_pred, cereal.y_test.numpy())
    print(f'mae: {mae_val:,.2f}')
