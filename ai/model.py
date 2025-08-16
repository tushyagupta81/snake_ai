import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.out(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, target_model, lr, gamma, update_freq=1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.target_update_freq = update_freq
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        self.train_step_count = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)   # integer indices
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        # Add batch dimension if single sample
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Q(s, a)
        pred = self.model(state)
        pred_expected = pred.gather(1, action.unsqueeze(1)).squeeze(1)

        # Q target
        with torch.no_grad():
            next_q_values = self.target_model(next_state).max(1)[0]  # max over actions
            target = reward + self.gamma * next_q_values * (~done)

        loss = self.criterion(pred_expected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
