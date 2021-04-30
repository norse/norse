# Parts of this code were adapted from the pytorch example at
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
# which is licensed under the license found in LICENSE.cartpole

from argparse import ArgumentParser
import numpy as np
import random
import os

import torch

# pytype: disable=import-error
import gym

# pytype: enable=import-error

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import LIFRecurrentCell
from norse.torch.module.lsnn import LSNNRecurrentCell, LSNNParameters
from norse.torch.module.leaky_integrator import LILinearCell


class ANNPolicy(torch.nn.Module):
    def __init__(self):
        super(ANNPolicy, self).__init__()
        self.state_space = 4
        self.action_space = 2
        self.l1 = torch.nn.Linear(self.state_space, 128, bias=False)
        self.l2 = torch.nn.Linear(128, self.action_space, bias=False)
        self.dropout = torch.nn.Dropout(p=0.6)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.softmax(x)
        return x


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFRecurrentCell(
            2 * self.state_dim,
            self.hidden_features,
            p=LIFParameters(method="super", alpha=100.0),
        )
        self.dropout = torch.nn.Dropout(p=0.5)
        self.readout = LILinearCell(self.hidden_features, self.output_features)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages = torch.zeros(
            seq_length, batch_size, self.output_features, device=x.device
        )

        s1 = so = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif(x[ts, :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        p_y = torch.nn.functional.softmax(m, dim=1)
        return p_y


class LSNNPolicy(torch.nn.Module):
    def __init__(self, model="super"):
        super(LSNNPolicy, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        # self.affine1 = torch.nn.Linear(self.state_dim, self.input_features)
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif_layer = LSNNRecurrentCell(
            2 * self.state_dim,
            self.hidden_features,
            p=LSNNParameters(method=model, alpha=100.0),
        )
        self.dropout = torch.nn.Dropout(p=0.5)
        self.readout = LILinearCell(self.hidden_features, self.output_features)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        # state for hidden layer
        s1 = None
        # state for output layer
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, self.output_features, device=x.device
        )

        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif_layer(x[ts, :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        p_y = torch.nn.functional.softmax(m, dim=1)
        return p_y


def select_action(state, policy, device):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, gamma):
    eps = np.finfo(np.float32).eps.item()
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.as_tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main(args):
    running_reward = 10
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    label = f"{args.policy}-{args.model}-{args.random_seed}"
    os.makedirs(f"runs/cartpole/{label}", exist_ok=True)
    os.chdir(f"runs/cartpole/{label}")
    with open("flags.txt", "w") as fp:
        fp.write(str(args))

    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    device = torch.device(args.device)

    env = gym.make(args.environment)
    env.reset()
    env.seed(args.random_seed)

    if args.policy == "ann":
        policy = ANNPolicy().to(device)
    elif args.policy == "snn":
        policy = Policy().to(device)
    elif args.policy == "lsnn":
        policy = LSNNPolicy(model=args.model).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    running_rewards = []
    episode_rewards = []

    for e in range(args.episodes):
        state, ep_reward = env.reset(), 0

        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state, policy, device=device)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy, optimizer, args.gamma)

        if e % args.log_interval == 0:
            print(
                "Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    e, args.episodes, ep_reward, running_reward
                )
            )
        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        if running_reward > env.spec.reward_threshold:
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, t)
            )
            break

    np.save("running_rewards.npy", np.array(running_rewards))
    np.save("episode_rewards.npy", np.array(episode_rewards))
    torch.save(optimizer.state_dict(), "optimizer.pt")
    torch.save(policy.state_dict(), "policy.pt")


if __name__ == "__main__":
    parser = ArgumentParser("Cartpole reinforcement learning task. Requires OpenAI gym")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use by pytorch.",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training trials."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor to use"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="In which intervals to display learning progress.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="super",
        choices=["super"],
        help="Model to use for training.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="snn",
        choices=["snn", "lsnn", "ann"],
        help="Select policy to use.",
    )
    parser.add_argument(
        "--render", type=bool, default=False, help="Render the environment"
    )
    parser.add_argument(
        "--environment", type=str, default="CartPole-v1", help="Gym environment to use."
    )
    parser.add_argument(
        "--random-seed", type=int, default=1234, help="Random seed to use"
    )
    main(parser.parse_args())
