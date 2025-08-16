import random
from collections import deque

import numpy as np
import torch

from ai.model import Linear_QNet, QTrainer
from graph.plot import plot_score
from snake_game.game import Direction, Point, SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)

        self.policy_net = Linear_QNet(23, 3)
        self.frozen_net = Linear_QNet(23, 3)
        self.trainer = QTrainer(self.policy_net, self.frozen_net, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # One step in each direction
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # ---------- Distance to wall in direction relative to heading ----------
        if dir_l:
            dist_s = head.x / 20 / (640 / 20)
            dist_r = head.y / 20 / (480 / 20)
            dist_l = (480 - head.y) / 20 / (480 / 20)
        elif dir_r:
            dist_s = (640 - head.x) / 20 / (640 / 20)
            dist_r = (480 - head.y) / 20 / (480 / 20)
            dist_l = head.y / 20 / (480 / 20)
        elif dir_u:
            dist_s = head.y / 20 / (480 / 20)
            dist_r = (640 - head.x) / 20 / (640 / 20)
            dist_l = head.x / 20 / (640 / 20)
        else:  # dir_d
            dist_s = (480 - head.y) / 20 / (480 / 20)
            dist_r = head.x / 20 / (640 / 20)
            dist_l = (640 - head.x) / 20 / (640 / 20)

        # ---------- Distance to nearest body segment ----------
        def distance_to_body(direction):
            dx, dy = direction
            steps = 1
            x, y = head.x + dx, head.y + dy
            while 0 <= x < 640 and 0 <= y < 480:
                if Point(x, y) in game.snake[1:]:
                    break
                x += dx
                y += dy
                steps += 1
            max_steps = (640 // 20) if dx != 0 else (480 // 20)
            return steps / max_steps

        # Straight / Right / Left directions based on heading
        if dir_u:
            dist_body_s = distance_to_body((0, -20))
            dist_body_r = distance_to_body((20, 0))
            dist_body_l = distance_to_body((-20, 0))
        elif dir_d:
            dist_body_s = distance_to_body((0, 20))
            dist_body_r = distance_to_body((-20, 0))
            dist_body_l = distance_to_body((20, 0))
        elif dir_l:
            dist_body_s = distance_to_body((-20, 0))
            dist_body_r = distance_to_body((0, -20))
            dist_body_l = distance_to_body((0, 20))
        else:  # dir_r
            dist_body_s = distance_to_body((20, 0))
            dist_body_r = distance_to_body((0, 20))
            dist_body_l = distance_to_body((0, -20))

        state = [
            # Distances to wall relative to heading
            dist_s, dist_r, dist_l,

            # Danger flags (immediate collision)
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down

            # Distance to fruit
            abs(game.food.x - head.x) / 20 / (640 / 20),
            abs(game.food.y - head.y) / 20 / (480 / 20),

            # Distance to wall absolute
            head.y / 20 / (480 / 20),
            (480 / 20 - head.y / 20) / (480 / 20),
            head.x / 20 / (640 / 20),
            (640 / 20 - head.x / 20) / (640 / 20),

            # Distance to nearest body segment in S, R, L
            dist_body_s, dist_body_r, dist_body_l
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        action_idx = torch.tensor([action.index(1)])
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_long_mem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_mem(self, state, action, reward, next_state, done):
        action_idx = action.index(1)
        self.trainer.train_step(state, action_idx, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            pred = self.policy_net(state0)
            move = int(torch.argmax(pred).item())

        final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_mem(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            # go over the whole history of the game
            agent.train_long_mem()
            agent.n_games += 1
            game.reset()

            if score > record:
                record = score
                agent.policy_net.save("online_model.pth")
                agent.frozen_net.save("frozen_model.pth")

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_score(plot_scores, plot_mean_scores)
