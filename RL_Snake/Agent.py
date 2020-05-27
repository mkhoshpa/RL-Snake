from .Game import Game
from .helper import plot_durations

from collections import namedtuple
from itertools import count
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
ACTIONS = [UP,DOWN,LEFT,RIGHT]


BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 150
TARGET_UPDATE = 5
MODEL_DIR = '../model'
num_episodes = 10000

class BaseAgent:
    """
    Abstract class for the snake player agent
    """
    def __init__(self, state=None):
        """
        constructor of the agent
        Args:
            state: optional: if is None means the start of the game, else shows the current state of the game
        """
        self.state = state if state is not None else (None,None)

    def update(self,board):
        """

        Args:
            board:

        Returns:

        """
        self.state = self.state[1], board

    def take_action(self):
        """
        Abstract method for taking the action by the agent according to it's state
        Returns:

        """
        pass

    def get_state(self):
        if self.state[0] is None or self.state[1] is None:
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(np.stack(self.state),device=device)


class RandomAgent(BaseAgent):
    """
    Concrete agent class that take actions randomly
    """
    def take_action(self):
        """
        take actions randomly
        Returns:
            str or None: represent the action
        """
        if self.state[0] is None or self.state[1] is None:
            return None
        return ACTIONS[random.randrange(len(ACTIONS))]


class DQN(nn.Module):
        """

        """
        def __init__(self, h, w, outputs):
            """

            Args:
                h: height of the board
                w: width of the board
                outputs: number of actions
            """
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=3, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            """

            Args:
                x: Tensor representation of input states

            Returns:
                list of int: representing the Q values of each state-action pair
            """
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))


class DQNAgent(BaseAgent):

    def __init__(self,board_size=(20,20),path=None):
        """
        An agent based on DQN. uses policy_net to select an action
        Args:
            board_size: tuple: height and width of the board
            trained: if True, the agent is already trained and there is no need to train the agent
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = len(ACTIONS) + 1
        self.board_size = board_size
        self.policy_net = DQN(board_size[0],board_size[1],outputs=self.n_actions)
        self.target_net = DQN(board_size[0], board_size[1], outputs=self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.steps_done = 0
        self.transitions = namedtuple('Transition',
                                      ('state', 'action', 'next_state', 'reward'))
        self.trained = False
        if path is None:
            self.train()
            self.trained = False
        else:
            self.policy_net.load_state_dict(torch.load(path))
            self.trained = True


    def take_action(self):
        """
        select an action based on the given state
        Args:
            None
        Returns:
            action: int.
        """
        state = self.get_state()
        if state is None:
            return torch.tensor([4])
        # make sure state is in the form of B*C*W*H
        if len(state.shape) == 3:
            state = state.unsqueeze(0).float()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if self.trained or sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                out = self.policy_net(state).max(1)[1].view(1, 1)
                return out
        else:
            out =  torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
            return out

    def _optimize_model(self):
        """
        trains the policy_net for one batch.

        Returns:
            None
        """
        if len(self._memory) < BATCH_SIZE:
            return
        transitions = self._memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.transitions(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def train(self):
        """
            trains the policy_net from scratch for fixed number of episodes.
        Returns:
            None
        """
        #named tuple rpresenting the transition ('state', 'action') -> 'next_state', 'reward'
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

        # Replay memory class to store the previos transition and ro sample for the training
        class ReplayMemory(object):

            def __init__(self, capacity):
                self.capacity = capacity
                self.memory = []
                self.position = 0

            def push(self, *args):
                """Saves a transition."""
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = Transition(*args)
                self.position = (self.position + 1) % self.capacity

            def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

            def __len__(self):
                return len(self.memory)

        self._optimizer = optim.Adam(self.policy_net.parameters())
        self._memory = ReplayMemory(20000)

        episode_rewards = []
        #start training

        for i_episode in range(num_episodes):
            # Initialize the environment and state
            game = Game(board_size=self.board_size, agent=self)
            state = self.get_state()
            for t in count():
                # Select and perform an action
                old_state = self.get_state()
                action = game.receive_action()
                reward = game.one_time_step()
                done = True if reward == -10 else False
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                new_state = self.get_state()

                # Store the transition in memory
                if old_state is not None and new_state is not None:
                    self._memory.push(old_state, torch.tensor(action,device=self.device), new_state, reward)

                # Perform one step of the optimization (on the target network)
                self._optimize_model()
                if done:
                    episode_rewards.append(game.cur_reward)
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Plot the game score history every 100 episodes
            if i_episode % 100 == 0:
                plot_durations(episode_rewards)
                if not os.path.exists(MODEL_DIR):
                    os.mkdir(MODEL_DIR)
                torch.save(self.policy_net.state_dict(),MODEL_DIR+'/policy_net-'+str(i_episode)+'.pth')
        plot_durations(episode_rewards)