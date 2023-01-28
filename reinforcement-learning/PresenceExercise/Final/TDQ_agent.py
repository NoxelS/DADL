
import cv2
import time
import numpy as np
from snake_game import Snake_game
from random import sample
from collections import deque
import matplotlib.pyplot as plt

# The Agent that is trained in the training environment later.
class Agent:
	def __init__(self, state_dim, action_dim):
		self.q_table = np.zeros((state_dim, action_dim))

	def load(self, name):
		self.q_table = np.load(name + ".npy")
	
	def save(self, name):
		np.save(name, self.q_table)
	
	# gets next action according to epsilon-greedy policy
	def get_next_action(self, state, epsilon=0.2):
		if np.random.uniform(0,1) > epsilon:
			return np.argmax(self.q_table[state])
		else:
			return np.random.randint(4)
	
	# updates Q-table according to TD-Formula
	def fit(self, state, action, next_state, next_action, reward, gamma=0.9, alpha=0.2):
		error = (reward + gamma * self.q_table[next_state][next_action] - \
                 self.q_table[state][action])
		self.q_table[state][action] += alpha * error

# Training environment in which the agent can train
class TrainingEnv:
	def __init__(self, game, agent, episodes, state_reduction_function=(lambda x : x), reward_per_eps=[], length_per_eps=[]):
		self.game = game
		self.agent = agent
		self.episodes = episodes
		# saves rewards and snake lengths for all episodes
		self.reward_per_eps = reward_per_eps
		self.length_per_eps = length_per_eps
		# This is a function to change the state received from the game in post. In our case, state reduction is already
		# done in the game (for runtime reasons), and this function just flattens the matrix so the state is given as
		# vector
		self.state_reduction_function = state_reduction_function
		# saves the snake starting position for the reset later
		self.starting_position = self.game.snake_position.copy()

	# The actual method for training, with parameters gamma, starting epsilon (epsilon), ending epsilon (epsilon_min)
	# and epsilon goes from epsilon to epsilon_min over num_epsilon steps. T can constrain the timesteps before a game
	# is finished (in our case we don't really do that though)
	def train(self, gamma, epsilon, num_epsilons, epsilon_min, T=10000):
		epsilon_red = (epsilon - epsilon_min) / num_epsilons

		for e in range(self.episodes):

			reward_sum = 0
			self.game.reset(self.starting_position.copy(), [1,1])
			self.game.create_random_food()
			print("reset game, epsilon = ",end='')
			print(epsilon)
				
			game_end = False
			state = self.state_reduction_function(self.game.get_game_state())
			self.game.render()
			cv2.waitKey(100)

			if epsilon > epsilon_min:
				epsilon = epsilon - epsilon_red

			if epsilon < epsilon_min:
				epsilon = epsilon_min

			game_step = 0
			
			while not game_end and game_step < T:
				game_step += 1

				action = self.agent.get_next_action(state[0], epsilon)
				#print(action)

				reward = self.game.do_action(action)
				self.game.render()
				cv2.waitKey(100)
				
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
							
	
#print("Hello")

field_size = 12

# boundaries at edges of board
bounds = np.array([[x,0] for x in range(field_size)] + [[0,y] for y in range(1,field_size)] + \
	[[field_size-1,y] for y in range(1,field_size)] + [[x,field_size-1] for x in range(1,field_size-1)])

#print(bounds)
#bounds = np.array([[3,3], [2,2], [2,3], [3,2]])
#bounds = np.array([])
snake_pos = np.array([[5,5]])

sgame = Snake_game((field_size,field_size), bounds, snake_pos, 0, [1,1])

#start_game(sgame)
#sgame.render()

#cv2.waitKey(1000)
#cv2.destroyWindow('Snake')

sagent = Agent(sgame.state_space_dim, 4)

#sagent.load('trained_nets/6x6_walls_CubeCenter_128_2/trained_network')
sagent.load('trained_table')
#print(sagent.q_function.S(-10000))

# rew_per_eps = []
# len_per_eps = []
rew_per_eps = [x for x in np.load("rewards_per_eps.npy")]
len_per_eps = [x for x in np.load("length_per_eps.npy")]
#rew_per_eps = [x for x in np.load("other_data/rewards_per_eps_6x6woborder.npy")]
#rew_per_eps = [x for x in np.load("other_data/rewards_per_eps_6x6centercubeborder_1.npy")]
trainenv = TrainingEnv(sgame, sagent, 100000, \
	lambda x : (int(sum([np.exp2(i) * x[0][i] for i in range(len(x[0]))])), x[1]), \
	rew_per_eps, len_per_eps)

#trainenv = TrainingEnv(sgame, sagent, 100000, 12)
rew_avg = [sum(rew_per_eps[i*100:(i+1)*100]) / 100 for i in range(len(rew_per_eps) // 100)]

#print(sagent.q_function.S(-1))
#plt.plot(range(len(rew_per_eps) // 100), rew_avg)
#plt.show()

trainenv.train(0.95, 1, 10, 0.01, 100000)
