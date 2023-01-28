import cv2
import numpy as np
import time
from snake_game import Snake_game as Snake
from random import sample


class Agent:
    def __init__(self, state_dim, action_div):
        self.q_table = np.zeros((state_dim, action_div))
    
    def load(self, name):
        self.q_table = np.load(name + '.npy')
    
    def save(self, name):
        np.save(name, self.q_table)

    # epsilon-greedy policy
    def get_next_action(self, state, epsilon=.2):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state])
        
    def fit(self, state, action, next_state, next_action, reward, gamma=0.9, alpha=0.2):
        error = (reward + gamma * self.q_table[next_state, next_action]) - self.q_table[state, action]
        self.q_table[state, action] += alpha * error

class TrainingEnv:
    def __init__(self, game, agent, episodes, state_reduction_function=(lambda x : x), reward_per_eps=[], length_per_eps=[]):
        self.game = game
        self.agent = agent
        self.episodes = episodes
        self.state_reduction_function = state_reduction_function
        self.reward_per_eps = reward_per_eps
        self.length_per_eps = length_per_eps
        self.starting_postion = self.game.snake_position.copy()

    def train(self,  gamma, epsilon, num_epsilon, epsilon_min, T=10000):
        epsilon_red = (epsilon - epsilon_min) / num_epsilon

        for e in range(epsilon_red):
            self.game.reset(self.starting_postion.copy(), [1,1])
            self.game_random_food()
            print(f"reset game, epsilon={epsilon}")

            game_end = False
            state = self.state_reduction_function(self.game.get_game_state())
            self.game.render()
            cv2.waitKey(1)

            if epsilon > epsilon_min:
                epsilon -= epsilon_red

            game_step = 0

            while not game_end and game_step < T:
                game_step += 1
                action = self.agent.get_next_action(state[0], epsilon)
                reward = self.game.do_action(action)
                self.game.render()
                cv2.waitKey(1)
                next_state = self.state_reduction_function(self.game.get_game_state())
                next_action = self.agent.get_next_action(next_state[0], epsilon)
                alpha = 0.01
                self.agent.fit(state[0], action, next_state[0], next_action, reward, 0.95, alpha)
                reward_sum += reward
                game_end = next_state[1]
                state = next_state

        # save all the things (rewards, lengths per episode and trained network)
        self.reward_per_eps.append(reward_sum)	
        self.length_per_eps.append(self.game.snake_position.shape[0])
        print("reward episode " + str(e) + " is ",end='')
        print(reward_sum)       
        print("snake length episode " + str(e) + " is ",end='')
        print(self.length_per_eps[e])       
        np.save("rewards_per_eps", np.array(self.reward_per_eps))
        np.save("length_per_eps", np.array(self.length_per_eps))        
        self.agent.save("trained_table")

field_size = 12

bounds = np.array([[x,0] for x in range(field_size)] + [[0,y] for y in range(1,field_size)] + \
    [[field_size-1,y] for y in range(1,field_size)] + [[x,field_size-1] for x in range(1,field_size-1)])

snake_pos = np.array([[5,5]])
sgame = Snake((field_size, field_size), bounds, snake_pos, 0, [1,1])
sagent = Agent(sgame.state_space_dim, 4)

rew_per_eps = []
len_per_eps = []

trainenv = TrainingEnv(sgame, sagent, 100000, \
    lambda x : (int(sum([np.exp2(i) * x[0][i] for i in range(len(x[0]))])), x[1]), \
    rew_per_eps, len_per_eps)

trainenv.train(0.95, 1, 1600, 0.01, 10000)