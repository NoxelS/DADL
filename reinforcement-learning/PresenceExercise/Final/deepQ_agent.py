
import cv2
import time
import numpy as np
from snake_game import Snake_game
from random import sample
from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# transition (s_t,a_t,r_t,s_{t+1})
class Transition:
	# state is vector of the board
	def __init__(self, state, action, reward, next_state):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state

# the replay memory used in experience replay technique. It stores transitions. Uses deque for convenience.
class ReplayMemory:
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)
	
	def push(self, transition):
		self.memory.append(transition)

	def sample(self, batch_size):
		return sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

# The Agent that is trained in the training environment later. Uses neural network implemented in tensorflow.
class Agent:
	def __init__(self, dims, gamma_nn=0.00025, memory_size=100000):
		#self.q_function = Network(dims, gamma_nn)
		self.q_function = Sequential()
		for i in range(1,len(dims)-1):
			if i == 1:
				self.q_function.add(Dense(dims[i], input_shape=(dims[0],), activation='relu'))
			else:
				self.q_function.add(Dense(dims[i], activation='relu'))
		self.q_function.add(Dense(dims[-1], activation='softmax'))
		self.q_function.compile(loss='mse', optimizer=Adam(lr=gamma_nn))

		self.memory = ReplayMemory(memory_size)

	def load(self, name):
		self.q_function = keras.models.load_model(name)
	
	# gets next action according to epsilon-greedy policy
	def next_action(self, state, epsilon=0.2):
		q_values = self.q_function.predict(tf.expand_dims(tf.constant(state), axis=0), verbose=0)[0]
		if np.random.uniform(0,1) > epsilon:
			return np.argmax(q_values)
		else:
			return np.random.randint(4)
	
	# The actual fit step used in the deep Q-algorithm. Neural net is fitted to a minibatch of transitions, using the
	# (Bellman) formula for the Q-Updates
	def fit(self, batch_size_agent, gamma_agent):
		minibatch = self.memory.sample(batch_size_agent)
		states = np.array([t.state[0] for t in minibatch])
		actions = np.array([t.action for t in minibatch])
		rewards = np.array([t.reward for t in minibatch])
		next_states = np.array([t.next_state[0] for t in minibatch])
		dones = np.array([t.next_state[1] for t in minibatch])

		states = np.squeeze(states)
		next_states = np.squeeze(next_states)

		# target is reward if done is true (terminal state) and Bellman formula otherwise
		targets = rewards + gamma_agent*(np.amax(self.q_function.predict_on_batch(next_states), axis=1))*(1-dones)
		targets_full = self.q_function.predict_on_batch(states)

		ind = np.array([i for i in range(batch_size_agent)])
		targets_full[[ind], [actions]] = targets

		self.q_function.fit(states, targets_full, epochs=1, verbose=0)

# Training environment in which the agent can train
class TrainingEnv:
	def __init__(self, game, agent, episodes, batch_size, state_reduction_function=(lambda x : x), reward_per_eps=[], length_per_eps=[]):
		self.game = game
		self.agent = agent
		self.episodes = episodes
		self.batch_size = batch_size
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
			cv2.waitKey(1)

			if epsilon > epsilon_min:
				epsilon = epsilon - epsilon_red

			if epsilon < epsilon_min:
				epsilon = epsilon_min

			game_step = 0
			
			while not game_end and game_step < T:
				game_step += 1

				action = self.agent.next_action(state[0], epsilon)
				#print(action)

				reward = self.game.do_action(action)
				self.game.render()
				cv2.waitKey(1)
				
				next_state = self.state_reduction_function(self.game.get_game_state())

				# save transition in replay memory
				self.agent.memory.push(Transition(state, action, reward, next_state))

				game_end = next_state[1]
				if(game_end):
					print("you died")

				if game_step == T and not next_state[1]:
					next_state = (next_state[0],True)
					reward -= 5
					print("last step")

				reward_sum += reward

				state = (next_state[0].copy(), next_state[1])

				if len(self.agent.memory) >= self.batch_size:
					self.agent.fit(self.batch_size, gamma)

			# save all the things (rewards, lengths per episode and trained network)
			self.reward_per_eps.append(reward_sum)	
			self.length_per_eps.append(self.game.snake_position.shape[0])
			print("reward episode " + str(e) + " is ",end='')
			print(reward_sum)

			print("snake length episode " + str(e) + " is ",end='')
			print(self.game.snake_position.shape[0])

			np.save("rewards_per_eps", np.array(self.reward_per_eps))
			np.save("length_per_eps", np.array(self.length_per_eps))

			self.agent.q_function.save("trained_network")
							
	
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

sagent = Agent([sgame.get_game_state()[0].flatten().shape[0], 128, 128, 128, 4], 0.00025)

#sagent.load('trained_nets/6x6_walls_CubeCenter_128_2/trained_network')
#sagent.load('trained_network')
#print(sagent.q_function.S(-10000))

rew_per_eps = []
len_per_eps = []
#rew_per_eps = [x for x in np.load("rewards_per_eps.npy")]
#len_per_eps = [x for x in np.load("length_per_eps.npy")]
#rew_per_eps = [x for x in np.load("other_data/rewards_per_eps_6x6woborder.npy")]
#rew_per_eps = [x for x in np.load("other_data/rewards_per_eps_6x6centercubeborder_1.npy")]
trainenv = TrainingEnv(sgame, sagent, 100000, 400, lambda x : (x[0].flatten(), x[1]), rew_per_eps, len_per_eps)

#trainenv = TrainingEnv(sgame, sagent, 100000, 12)
rew_avg = [sum(rew_per_eps[i*100:(i+1)*100]) / 100 for i in range(len(rew_per_eps) // 100)]

#print(sagent.q_function.S(-1))
#plt.plot(range(len(rew_per_eps) // 100), rew_avg)
#plt.show()

trainenv.train(0.95, 1, 100, 0.01, 100000)
	
	
		
		
