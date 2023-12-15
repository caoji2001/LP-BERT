# Human Mobility Prediction Challenge: Next Location Prediction using Spatiotemporal BERT

## Overview

This repository contains an **unofficial** PyTorch implementation of the ["Human Mobility Prediction Challenge: Next Location Prediction using Spatiotemporal BERT"](https://dl.acm.org/doi/10.1145/3615894.3628498) method, as part of the [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023).

## Setup

```bash
pip install -r requirements.txt
```

## Run

1. Prepare data

Download the data from [here](https://zenodo.org/records/10142719) and place it in the `data` directory.

2. Train

```bash
python train_task1.py --batch_size 128 --epochs 200 --embed_size 128 --layers_num 4 --heads_num 8

python train_task2.py --batch_size 128 --epochs 200 --embed_size 128 --layers_num 4 --heads_num 8
```

3. Predict

Here, `${PTH_FILE_PATH}` refers to the path where the PTH file obtained after training the corresponding task.

```bash
python val_task1.py --pth_file ${PTH_FILE_PATH} --embed_size 128 --layers_num 4 --heads_num 8

python val_task2.py --pth_file ${PTH_FILE_PATH} --embed_size 128 --layers_num 4 --heads_num 8
```

## License

This project is licensed under the [MIT License](https://github.com/caoji2001/Human-Mobility-Prediction-Challenge-Next-Location-Prediction-using-Spatiotemporal-BERT/blob/main/LICENSE).

## Citations

```bibtex
@inproceedings{10.1145/3615894.3628498,
author = {Terashima, Haru and Tamura, Naoki and Shoji, Kazuyuki and Katayama, Shin and Urano, Kenta and Yonezawa, Takuro and Kawaguchi, Nobuo},
title = {Human Mobility Prediction Challenge: Next Location Prediction Using Spatiotemporal BERT},
year = {2023},
isbn = {9798400703560},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3615894.3628498},
doi = {10.1145/3615894.3628498},
abstract = {Understanding, modeling, and predicting human mobility patterns in urban areas has become a crucial task from the perspectives of traffic modeling, disaster risk management, urban planning, and more. HuMob Challenge 2023 aims to predict future movement trajectories based on the past movement trajectories of 100,000 users[1]. Our team, "uclab2023", considered that model design significantly impacts training and prediction times in the task of human mobility trajectory prediction. To address this, we proposed a model based on BERT, commonly used in natural language processing, which allows parallel predictions, thus reducing both training and prediction times.In this challenge, Task 1 involves predicting the 15-day daily mobility trajectories of target users using the movement trajectories of 100,000 users. Task 2 focuses on predicting the 15-day emergency mobility trajectories of target users with data from 25,000 users. Our team achieved accuracy scores of GEOBLEU: 0.3440 and DTW: 29.9633 for Task 1 and GEOBLEU: 0.2239 and DTW: 44.7742 for Task 2[2][3].},
booktitle = {Proceedings of the 1st International Workshop on the Human Mobility Prediction Challenge},
pages = {1–6},
numpages = {6},
keywords = {transformer, human mobility, next location prediction},
location = {Hamburg, Germany},
series = {HuMob-Challenge '23}
}
```
