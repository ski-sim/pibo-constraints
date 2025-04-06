import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll
from tqdm import tqdm
import wandb  # wandb import 추가
from model import Flow
import argparse
import yaml
from util import sample_visual
# wandb 초기화
wandb.init(project="PIBO", name="Flow-matching")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_sample_size', type=int, default=256)
    parser.add_argument('--test_sample_size', type=int, default=300)
    parser.add_argument('--step_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--dataset', type=str, choices=["moons", "swiss"], default="moons")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--config', default=None, type=str, help='path to config file,'
                        ' and command line arguments will be overwritten by the config file')
    
    args = parser.parse_args()

    if args.config:
        with open("config.yaml", "r") as f:
            cfg_dict = yaml.safe_load(f)

            for key, val in cfg_dict.items():
                assert hasattr(args, key), f'Unknown config key: {key}'
                setattr(args, key, val)
            f.seek(0)
            print(f'Config file: {args.config}', )
            for line in f.readlines():
                print(line.rstrip())

    return args


if __name__ == '__main__':

    args = parse_arguments()
    # training
    flow = Flow()
    optimizer = torch.optim.Adam(flow.parameters(), args.learning_rate)
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(args.epochs)):
        train_data = make_moons(args.train_sample_size, noise=0.05)[0] # shape : (count, dim)
        x_1 = Tensor(train_data)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1)

        x_t = (1-t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        optimizer.zero_grad()
        loss = loss_fn(flow(x_t, t), dx_t)
        loss.backward()
        
        # logging
        wandb.log({"loss": loss.item(), "epoch": _})
        optimizer.step()

    # save model
    torch.save(flow.state_dict(), "./result/flow_model.pth")

    # sampling visualization
    sample_visual(flow, args.test_sample_size, train_data.shape[1], args.step_size)

    wandb.finish()