import os
import argparse
import logging
import random
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from model import *


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    len_tensor = torch.tensor([item['len'] for item in batch])

    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)

    return {
        'd': d_padded,
        't': t_padded,
        'input_x': input_x_padded,
        'input_y': input_y_padded,
        'time_delta': time_delta_padded,
        'label_x': label_x_padded,
        'label_y': label_y_padded,
        'len': len_tensor
    }


def task2(args):
    name = f'batchsize{args.batch_size}_epochs{args.epochs}_embedsize{args.embed_size}_layersnum{args.layers_num}_headsnum{args.heads_num}_cuda{args.cuda}_lr{args.lr}_seed{args.seed}'
    current_time = datetime.datetime.now()

    log_path = os.path.join('log', 'task2', name)
    tensorboard_log_path = os.path.join('tb_log', 'task2', name)
    checkpoint_path = os.path.join('checkpoint', 'task2', name)

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.txt'),
                        filemode='w')
    writer = SummaryWriter(tensorboard_log_path)

    task2_dataset_train = HuMobDatasetTask2Train('./data/task2_dataset_kotae.csv')
    task2_dataloader_train = DataLoader(task2_dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    device = torch.device(f'cuda:{args.cuda}')
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch_id in range(args.epochs):
        for batch_id, batch in enumerate(tqdm(task2_dataloader_train)):
            batch['d'] = batch['d'].to(device)
            batch['t'] = batch['t'].to(device)
            batch['input_x'] = batch['input_x'].to(device)
            batch['input_y'] = batch['input_y'].to(device)
            batch['time_delta'] = batch['time_delta'].to(device)
            batch['label_x'] = batch['label_x'].to(device)
            batch['label_y'] = batch['label_y'].to(device)
            batch['len'] = batch['len'].to(device)

            output = model(batch['d'], batch['t'], batch['input_x'], batch['input_y'], batch['time_delta'], batch['len'])
            label = torch.stack((batch['label_x'], batch['label_y']), dim=-1)

            pred_mask = (batch['input_x'] == 201)
            pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)

            loss = criterion(output[pred_mask], label[pred_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step = epoch_id * len(task2_dataloader_train) + batch_id
            writer.add_scalar('loss', loss.detach().item(), step)
        scheduler.step()

        logging.info(f'epoch: {epoch_id}, loss: {loss.detach().item()}')

    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    task2(args)
