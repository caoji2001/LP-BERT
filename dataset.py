import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class HuMobDatasetTask1Train(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []

        for uid, traj in tqdm(self.df.groupby('uid')):
            if uid >= 80000:
                traj = traj[traj['d'] < 60]

            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }


class HuMobDatasetTask1Val(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []

        self.df = self.df[self.df['uid'] >= 80000]
        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }


class HuMobDatasetTask2Train(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []

        for uid, traj in tqdm(self.df.groupby('uid')):
            if uid >= 22500:
                traj = traj[traj['d'] < 60]

            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            d_unique = np.unique(d)
            if len(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)]) == 0:
                continue
            mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & (d_unique <= np.max(d_unique) - 14)])
            mask_d_end = mask_d_start + 14
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }


class HuMobDatasetTask2Val(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []

        self.df = self.df[self.df['uid'] >= 22500]
        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])

        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }
