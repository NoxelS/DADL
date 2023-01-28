
import cv2
import time
import numpy as np

# returns L1 distance between 2 2D-points, used for reward shaping
def distance(point1, point2):
    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

# defines the snake game: state is ((array for snake position, snake head direction), food position)
# directions: 0 right, 1 up, 2 left, 3 down

class Snake_game:
	#state = np.array[[[[0,0]],0],[0,0]]
	#board_size = (0,0)
	#boundaries = np.array(0)
	#snake_position = np.array([[0,0]])
	#snake_direction = 0
	#food_position = np.array([-1,-1])
	#board_matrix = np.zeros((board_size[0], board_size[1], 3))

	def __init__(self, board_s, bounds=np.array(0), snake_pos=np.array([[0,0]]), snake_dir=0, food_pos=np.array([-1,-1]), \
		food_reward=10, reinforce=1, death=100):
		self.board_size = board_s
		self.boundaries = bounds
		self.snake_position = snake_pos
		self.snake_direction = snake_dir
		self.food_position = food_pos
		self.board_matrix = np.zeros((self.board_size[0], self.board_size[1], 3))
		self.has_ended = False
		self.food_reward = food_reward
		self.reinforce = reinforce
		self.death = death
		self.state_space_dim = int(np.exp2(self.get_game_state()[0].size))
		if self.boundaries.size != 0:
			for pos in range(self.boundaries.shape[0]):
				self.board_matrix[self.boundaries[pos][0],self.boundaries[pos][1],0] = 255

	def get_game_state_bounds_snake(self):
		ret_mat = np.zeros((self.board_size[0], self.board_size[1], 3))
		for pos in range(1,self.snake_position.shape[0]):
			ret_mat[self.snake_position[pos][0],self.snake_position[pos][1], 0] = 1
		if self.boundaries.size != 0:
			for pos in range(self.boundaries.shape[0]):
				ret_mat[self.boundaries[pos][0],self.boundaries[pos][1], 0] = 1
		if self.food_position[0] != -1:
			ret_mat[self.food_position[0], self.food_position[1], 1] = 1
		ret_mat[self.snake_position[0][0], self.snake_position[0][1], 2] = 1

		return np.array([x for x in ret_mat.flatten()] + [self.snake_direction == 0, self.snake_direction == 1, self.snake_direction == 2, self.snake_direction == 3], dtype=int)

	# returns a vector of 3 boards, for snake, food and boundaries
	def get_game_state_complete(self):
		ret_mat = np.zeros((self.board_size[0], self.board_size[1], 3))
		for pos in range(self.snake_position.shape[0]):
			ret_mat[self.snake_position[pos][0],self.snake_position[pos][1], 0] = 1
		if self.food_position[0] != -1:
			ret_mat[self.food_position[0], self.food_position[1], 1] = 1
		if self.boundaries.size != 0:
			for pos in range(self.boundaries.shape[0]):
				ret_mat[self.boundaries[pos][0],self.boundaries[pos][1], 2] = 1

		return ret_mat

	# returns vector of (boundaries, snake position, food position, snake direction). The length varies with snake
	# length, so not really fit for evaluation functions 
	def get_game_state_nonuniform(self):
		return np.array([x for x in np.array([x for x in self.boundaries] + [x for x in self.snake_position] + [self.food_position]).flatten()] + [self.snake_direction])
	
	# gives reduced state vector in {0,1}^n:
	# (is food right of snake head, is food above snake head, is food left of snake head, is food below snake head,
	# obstacle right of snake, obstacle above snake, obstacle left of snake, obstacle below snake,
	# does snake look down, does snake look right, does snake look up, does snake look left)
	def get_game_state_reduced(self):
		ret_mat = np.array([self.food_position[0] > self.snake_position[0][0], self.food_position[1] > self.snake_position[0][1], \
			self.food_position[0] < self.snake_position[0][0], self.food_position[1] < self.snake_position[0][1], \
			np.any(np.all(np.equal(self.snake_position[1:], self.snake_position[0] + [1,0]),axis=1)), \
			np.any(np.all(np.equal(self.snake_position[1:], self.snake_position[0] + [0,1]),axis=1)), \
			np.any(np.all(np.equal(self.snake_position[1:], self.snake_position[0] - [1,0]),axis=1)), \
			np.any(np.all(np.equal(self.snake_position[1:], self.snake_position[0] - [0,1]),axis=1)), \
			self.snake_direction == 0, self.snake_direction == 1, self.snake_direction == 2, self.snake_direction == 3], dtype=int)
		return ret_mat
	
	# returns the matrix form of the game state in a tuple with a boolean value that tells if it is a terminal state
	def get_game_state(self):
		return (self.get_game_state_reduced(), self.has_ended)

	# returns the matrix form of the game state in a tuple with a boolean value that tells if it is a terminal state,
	# always reduced state for TDQ-Agent
	def get_game_state_td(self):
		return (self.get_game_state_reduced(), self.has_ended)

	# creates random food pellet on the board
	def create_random_food(self):
		self.food_position = np.random.randint(0,[self.board_size[0], self.board_size[1]], size=2)
		#self.food_position = np.array([1,1])
		while np.any(np.all(np.equal(self.snake_position, self.food_position),axis=1)) or \
			(self.boundaries.size != 0 and np.any(np.all(np.equal(self.boundaries, self.food_position),axis=1))):
			self.food_position = np.random.randint(0,[self.board_size[0], self.board_size[1]], size=2)

	# performs a move and returns a reward. self.reinforce is a parameter specifying the strength of the reward shaping,
	# and self.death specifies the penalty on death. self.food_reward is the reward on eating food pellet
	def move(self):
		reward = 0

		move_update = np.array([1 if self.snake_direction == 0 else (-1 if self.snake_direction == 2 else 0), \
			1 if self.snake_direction == 1 else (-1 if self.snake_direction == 3 else 0)])
		old_last_position = self.snake_position[-1].copy()
		prev_position = self.snake_position[0].copy()
		if self.snake_position.shape[0] > 1:
			old_position = self.snake_position[:-1].copy()
		self.snake_position[0] = self.snake_position[0] + move_update
		if self.snake_position.shape[0] > 1:
			self.snake_position[1:] = old_position

		self.snake_position[:,0] = self.snake_position[:,0] % self.board_size[0]
		self.snake_position[:,1] = self.snake_position[:,1] % self.board_size[1]

		if np.all(np.equal(self.snake_position[0], self.food_position)):
			self.snake_position.resize((self.snake_position.shape[0] + 1, self.snake_position.shape[1]), refcheck=False)
			self.snake_position[-1] = old_last_position
			self.create_random_food()
			reward += self.food_reward
		# reward shaping: if distance to food is reduced, add reward, if it is increased, add penalty
		elif distance(self.snake_position[0], self.food_position) < distance(prev_position, self.food_position):
			reward += self.reinforce
		elif distance(self.snake_position[0], self.food_position) > distance(prev_position, self.food_position):
			reward -= self.reinforce

		if np.any(np.all(np.equal(self.snake_position[1:], self.snake_position[0]),axis=1)):
			reward -= self.death
			self.has_ended = True

		if self.boundaries.size != 0:
			if np.any(np.all(np.equal(self.boundaries, self.snake_position[0]),axis=1)):
				reward -= self.death
				self.has_ended = True

		return reward

	# self-explanatory
	def change_direction(self, direction):
		ret = 0
		if abs(self.snake_direction - direction) != 2:
			self.snake_direction = direction
		else:
			ret = 0
		return ret
	
	#self-explanatory
	def do_action(self, action):
		ret = self.change_direction(action)
		ret += self.move()
		return ret
		#return self.change_direction(action) + self.move() 

	# returns all possible actions in the given game state, not used in current agent implementations
	def get_possible_actions(self):
		return np.array([x for x in range(4) if abs(self.snake_direction - x) != 2])
	
	# resets the game with a given snake position and food position
	def reset(self, snake_p, food_p):
		self.has_ended = False
		self.snake_position = snake_p
		self.food_position = food_p
		self.snake_direction = 0

	# ressets the board matrix for rendering
	def reset_board(self):
		self.board_matrix = np.zeros((self.board_size[0], self.board_size[1], 3))
		for pos in range(self.boundaries.shape[0]):
			self.board_matrix[self.boundaries[pos][0],self.boundaries[pos][1],0] = 255

	# creates a matrix and renders the game with cv2
	def render(self):
		self.reset_board()
		for pos in range(self.snake_position.shape[0]):
			self.board_matrix[self.snake_position[pos][0],self.snake_position[pos][1],1] = 255
		if self.food_position[0] != -1:
			self.board_matrix[self.food_position[0], self.food_position[1], 2] = 255

		#cv2.imshow('Snake', np.array(self.board_matrix, dtype=np.uint8))

		cv2.namedWindow('Snake', cv2.WINDOW_NORMAL)
		#img = cv2.resize(self.board_matrix, dsize=(500,500))
		cv2.resizeWindow('Snake', 900, 900)
		cv2.imshow('Snake', self.board_matrix)

# can play the game normally for human players
def start_game(sgame1):
	sgame1.render()
	reset_positions = (sgame1.snake_position.copy(), sgame1.food_position.copy())

	rew = 0

	while not sgame1.get_game_state()[1]:
		p_key = cv2.waitKey(250)
		if p_key in [ord('w'), ord('a'), ord('s'), ord('d')]:
			sgame1.change_direction(0 if p_key == ord('s') else (1 if p_key == ord('d') else (2 if p_key == ord('w') else 3)))
		rew += sgame1.move()
		sgame1.render()

	print("points:",end='')
	print(rew + 50)
	cv2.destroyWindow('Snake')
	sgame1.reset(reset_positions[0].copy(), reset_positions[1].copy())

field_size = 12

bounds = np.array([[x,0] for x in range(field_size)] + [[0,y] for y in range(1,field_size)] + \
	[[field_size-1,y] for y in range(1,field_size)] + [[x,field_size-1] for x in range(1,field_size-1)])

snake_pos = np.array([[5,5]])

sgame = Snake_game((field_size,field_size), bounds, snake_pos, 0, [1,1])

start_game(sgame)
		
