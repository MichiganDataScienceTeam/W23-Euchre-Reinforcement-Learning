'''Euchre DMC
'''

import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    env = rlcard.make(args.env)

    # Initialize the DMC trainer
    trainer = DMCTrainer(env,
                         load_model=args.load_model,
                         xpid=args.xpid,
                         savedir=args.savedir,
                         save_interval=args.save_interval,
                         num_actor_devices=args.num_actor_devices,
                         num_actors=args.num_actors,
                         training_device=args.training_device)

    # Train DMC Agents
    trainer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--env', type=str, default='euchre')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
    parser.add_argument('--xpid', default='euchre',
                        help='Experiment id (default: doudizhu)')
    parser.add_argument('--savedir', default='experiments/dmc_result',
                        help='Root dir where experiment data will be saved')
    parser.add_argument('--save_interval', default=30, type=int,
                        help='Time interval (in minutes) at which to save the model')
    parser.add_argument('--num_actor_devices', default=1, type=int,
                        help='The number of devices used for simulation')
    parser.add_argument('--num_actors', default=4, type=int,
                        help='The number of actors for each simulation device')
    parser.add_argument('--training_device', default=0, type=int,
                        help='The index of the GPU used for training models')
    parser.add_argument('--total_frames', default=2000000, type=int,
                        help='Number of frames (timesteps) to run algorithm for')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)