import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearningTable:
    def __init__(self, actions, learning_rate=0.9, reward_decay=0.5, e_greedy=1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def get_q_table(self):
        return self.q_table

    def reduce_epsilon(self):
        self.epsilon -= 0.1

    def choose_action(self, state, avail_actions):
        self.check_state_exist(state)

        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[state, avail_actions]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.idxmax()
            if self.q_table.loc[state, action] == 0:
                action = np.random.choice(avail_actions)
        else:
            action = np.random.choice(avail_actions)

        return action

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)

        q_predict = self.q_table.loc[state, action]
        q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def plot_results(self, steps, cost):
        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Episode via reward')

        plt.tight_layout()

        # plt.figure()
        # plt.plot(np.arange(len(steps)), steps, 'b')
        # plt.title('Episode via steps')
        # plt.xlabel('Episode')
        # plt.ylabel('Steps')

        # plt.figure()
        # plt.plot(np.arange(len(cost)), cost, 'r')
        # plt.title('Episode via cost')
        # plt.xlabel('Episode')
        # plt.ylabel('Cost')

        plt.show()
