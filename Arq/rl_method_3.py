
# ============================================================================
# IMPORTS
# ============================================================================

from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
from numpy import random as rnd
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ============================================================================

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
print('Using device:', device)

# ============================================================================


# DQN:
class Environment():
    def __init__(self, estados, acciones):
        self.estados = estados
        self.acciones = acciones
        self.dim_estados = len(self.estados[0])
        self.num_acciones = len(self.acciones)
        self.reset()

    def reset(self):
        i = random.choice(range(len(self.estados)))
        self.state = self.estados[i]  # estado inicial random

    def execute_action(self, action):
        self.state = self.acciones[action]
        res = self.subscriber.update(self.state)
        self.reward = 1/res
        return self.reward


class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, state):
        qvals = self.ff(state)
        return qvals


class Buffer():
    def __init__(self):
        self.buffer = []

    def size(self):
        return len(self.buffer)

    def push(self, state, action, new_state, reward):
        experience = (state, action, new_state, reward)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batchSample = random.sample(self.buffer, batch_size)
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []

        for experience in batchSample:
            state, action, new_state, reward = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
        return (state_batch, action_batch, reward_batch, new_state_batch)


class DeepAgent():
    def __init__(self, dim_estados, num_acciones, episodios):
        self.policy_net = DQN(dim_estados, num_acciones).to(device)
        self.target_net = DQN(dim_estados, num_acciones).to(device)
        self.target_net.eval()
        self.target_update = 1000
        self.replay_buffer = Buffer()
        self.eps_start = 1
        self.eps_end = 0.1
        self.lim_episodes = 229
        self.eps_decay = 0.99
        self.epsilon = self.eps_start
        self.gamma = 0.1
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate)
        self.huber_loss = nn.MSELoss()

    def select_action(self, state, num_acciones):
        state = torch.FloatTensor(state).float().to(device)
        if rnd.rand() < (1-self.epsilon):
            with torch.no_grad():
                qvals = self.policy_net.forward(state)
                action = np.argmax(qvals.cpu().detach().numpy())
        else:
            action = random.choice(list(range(num_acciones)))
        return action

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        curr_Q = self.policy_net.forward(states).gather(
            1, actions.unsqueeze(1))
        next_Q = self.target_net.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q
        loss = self.huber_loss(curr_Q, expected_Q.unsqueeze(1))
        return loss


class DeepRLInterface():
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.batch_size = 50

    def step(self, num_acciones):
        state = self.env.state.copy()
        action = self.agent.select_action(state, num_acciones)
        rew = self.env.execute_action(action)
        new_state = self.env.state.copy()
        self.agent.replay_buffer.push(state, action, new_state, rew)
        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss)
        return state, action, rew, new_state

    def runTrials(self, nTrials, steps, num_acciones):
        counter = 0
        self.rewlist = []
        self.losslist = []
        for i in range(nTrials):
            self.env.reset()
            total_rew = 0
            for j in range(steps):
                state, action, rew, new_state = self.step(num_acciones)
                total_rew += rew
                print("Episodio:", i, "Paso:", j)
                print("Estado:", new_state, "Accion:", action)
                print("Recompensa:", rew, "Epsilon:", self.agent.epsilon)
                counter += 1
            self.rewlist.append(total_rew)

            if counter % self.agent.target_update == 0:
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict())

            self.agent.epsilon = self.agent.eps_start - (
                i * self.agent.eps_decay)

            self.writer.add_scalar('Recompensa acumulada por episodio',
                                   total_rew, i)
            self.writer.add_scalar('Epsilon', self.agent.epsilon, i)

            if (i+1) % 100 == 0:
                PATH = (f"model-12-{i}.pt")
                torch.save({
                        'epoch': i,
                        'model_state_dict': self.agent.target_net.state_dict(),
                        'optimizer_state_dict':
                        self.agent.optimizer.state_dict(),
                        }, PATH)


class RL_Method_3(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        # numero de episodios y pasos
        self.ep_maximo = 500
        self.t_maximo = 100

        # inicializar estados y acciones
        e1 = np.arange(60, 310, 10)
        e2 = np.repeat(e1, 25)
        e3 = np.arange(10, 60, 10)
        e4 = np.tile(e3, 125)
        e5 = np.arange(1, 6, 1)
        e6 = np.repeat(e5, 5)
        e7 = np.tile(e6, 25)
        e8 = np.column_stack((e2, e4))
        self.S = np.column_stack((e8, e7))  # 625 estados
        self.acciones = np.column_stack((e8, e7))  # 625 acciones

        # inicializar agente y entorno
        self.env = Environment(estados=self.S, acciones=self.acciones)
        self.n_estados = self.env.dim_estados
        self.n_acciones = self.env.num_acciones
        self.agente = DeepAgent(self.n_estados, self.n_acciones,
                                self.ep_maximo)

    # funcion RL- actualizar estados y matriz Q
    def process(self):
        self.writer = SummaryWriter()
        rl = DeepRLInterface(self.agente, self.env)
        rl.runTrials(self.ep_maximo, self.t_maximo, self.n_acciones)
