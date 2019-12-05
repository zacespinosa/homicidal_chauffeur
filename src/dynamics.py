import matplotlib.pyplot as plt
import numpy as np
import random as random

"""
Defines the Pursuer, Evader, and Simulation classes
By: Tarun Punnoose, Zac Espinosa
Last Edit: 11/11/19
"""
#############################################
#############################################
class Pursuer():
	def __init__(self, pos_p=np.zeros(3), w_p=2.0, a_min=-.5, a_max=.5, L=.3):
		"""
		Create pursuer

		param w_p: speed pursuer (m/s)
		param a_min: min action (-pi rads)
		param a_max: max action (pi rads)
		param L: distance between wheels (m)
		"""
		self.pos = pos_p
		self.s = None
		self.w = w_p
		self.pos_init = pos_p
		self.a_min = a_min
		self.a_max = a_max
		self.L = L
		self.R_p = L/np.tan(a_max)


		# ## Q-Learning
		# num_states = d_steps*phi_steps*phi_dot_steps
		# self.Q = np.random.random((num_states, a_steps))

	def update_state(self, s_p):
		self.s = s_p
		# print(s_p)

	def f_p(self, pos, a):
		"""
		Continuous dynamics of the pursuer
		"""
		s_dot = np.zeros(3)
		s_dot[0] = self.w*np.cos(pos[2])
		s_dot[1] = self.w*np.sin(pos[2])
		s_dot[2] = self.w/self.R_p*np.clip(a, self.a_min, self.a_max)

		return s_dot

	def optimal_strategy(self, pos_e, pos_p):
		"""
		Optimal strategy for the pursuer as obtained by Isaacs
		"""
		a_p = np.arctan2(pos_e[1] - pos_p[1], pos_e[0] - pos_p[0]) - pos_p[2]
		return a_p

	# def QLearning(self, sp, s, a):
	# 	"""
	# 	Simple Q-Learning
	# 	"""
	#
	# 	self.Q[s, a] = self.Q[s, a] + self.alpha*(r + self.gamma*np.max(Q[sp,:]) - Q[s, a])

#############################################
#############################################
class Evader():
	def __init__(self, d_steps, phi_steps, phi_dot_steps, a_steps, pos_e=np.zeros(2), w_e=1.0):
		"""
		Create evader

		param w_e: speed evader
		"""
		self.s = None
		self.w = w_e
		self.pos = pos_e
		self.pos_init = pos_e

		## Q-Learning
		self.num_states = d_steps*phi_steps*phi_dot_steps
		self.num_actions = a_steps
		# TODO: more intelligent initialization of Q
		self.Q = np.random.random((self.num_states, self.num_actions))
		self.gamma = 0.95
		self.alpha = 0.1
		self.eps = 0.0

	def update_state(self, s_e):
		self.s = s_e
		# print(s_e)

	def f_e(self, s, a):
		"""
		Continuous dynamics of the evader
		"""
		s_dot = np.zeros(2)
		s_dot[0] = self.w*np.cos(a)
		s_dot[1] = self.w*np.sin(a)

		return s_dot

	def optimal_strategy(self, pos_e, pos_p, R_p):
		"""
		Optimal strategy for the evader as obtained by Isaacs
		"""
		d = np.linalg.norm(pos_e - pos_p[0:2])

		if d > R_p:
			a_e = np.arctan2(pos_e[1] - pos_p[1], pos_e[0] - pos_p[0])
		else:
			a_e = pos_p[2] + np.pi/2

		return a_e

	def updateQ(self, sp, s, a, r):
		"""
		Simple Q-Learning
		"""
		self.Q[s, a] = self.Q[s, a] + self.alpha*(r + self.gamma*np.max(self.Q[sp,:]) - self.Q[s, a])

	def qLearningPolicy(self, s):
		"""
		ϵ-greedy exploration with Q learning based policy
		"""

		# don't explore
		if np.random.random() > self.eps:
			return np.argmax(self.Q[s,:])
		# explore with random action -> probably don't need to explore
		else:
			return np.random.randint(0, self.num_actions)



#############################################
#############################################
class Simulator():
	def __init__(self, p, e, d_steps, phi_steps, phi_dot_steps, a_steps, t=60, dt=0.05, step_r=1, end_r=10000, x_max=100, y_max=100, verbose=True):
		"""
		param p: instantiated pursuer
		param e: instantiated evader
		"""
		self.p = p
		self.e = e
		self.t = t
		self.dt = dt
		self.step_r = step_r # Reward at each time step
		self.end_r = end_r # Reward at end game
		self.x_max = x_max
		self.y_max = y_max

		self.curtime = 0
		self.capture_radius = self.p.L/2

		self.verbose = verbose
		if verbose:
			self.path = [[],[],[],[]]

		# Discrete state space
		d_upper = 30
		phi_lower = -4
		phi_upper = 4
		phi_dot_lower = -4
		phi_dot_upper = 4
		self.d_discrete = np.linspace(0, d_upper, d_steps)
		self.phi_discrete = np.linspace(phi_lower, phi_upper, phi_steps)
		self.phi_dot_discrete = np.linspace(phi_dot_lower, phi_dot_upper, phi_dot_steps)

		self.state_size_tuple = (d_steps, phi_steps, phi_dot_steps)

		# Discrete action space
		self.a_e_discrete = np.linspace(-np.pi, np.pi, a_steps)
		self.a_p_discrete = np.linspace(self.p.a_min, self.p.a_max, a_steps)

		# Set initial state of puruser and evader
		s_p, s_e = self.get_states(self.p.pos, self.e.pos)
		self.p.update_state(s_p)
		self.e.update_state(s_e)


	def increment_game(self, pos_p, pos_e):
		"""
		Increments the current time by one time step.
		"""
		self.curtime += self.dt
		if self.verbose:
			curtime = int(self.curtime/self.dt)
			self.path[0].append(pos_p[0])
			self.path[1].append(pos_p[1])
			self.path[2].append(pos_e[0])
			self.path[3].append(pos_e[1])

	def restart_game(self):
		"""
		Reset player positions and plot game
		This does not need to be called explicitly by user
		"""
		if self.verbose:
			fig, ax = plt.subplots()
			ax.plot(self.path[0], self.path[1], 'r--', label="car")
			ax.plot(self.path[2], self.path[3], 'b--', label="person")
			legend = ax.legend(loc='upper center', shadow=True, fontsize='large')
			plt.show()
			self.path = [[],[],[],[]]

		self.p.pos = self.p.pos_init
		self.e.pos = self.e.pos_init

	# def inbounds(self, s_p, s_e):
	# 	"""
	# 	Prevents agents from going off map
	# 	NOTE: I don't think we actually need this, but it's useful for now
	# 	NOTE: Looping to other side of map may be prefered

	# 	param s_p: pursuer state
	# 	param s_e: evader state
	# 	"""
	# 	p_x, p_y = s_p[0], s_p[1]
	# 	e_x, e_y = s_e[0], s_e[1]

	# 	if p_x < 0: p_x = 0
	# 	if p_x > self.x_max: p_x = self.x_max

	# 	if e_x < 0: e_x = 0
	# 	if e_x > self.x_max: e_x = self.x_max

	# 	if p_y < 0: p_y = 0
	# 	if p_y > self.y_max: p_y = self.y_max

	# 	if e_y < 0: e_y = 0
	# 	if e_y > self.y_max: e_y = self.y_max

	# 	return (np.array([p_x, p_y, s_p[2]]), np.array([e_x, e_y]))

	def get_reward(self, s_p, s_e, discrete_state=True):
		"""
		Returns reward for (s_p, s_e).

		param s_p: pursuer state
		param s_e: evader state
		"""
		if discrete_state:
			d_index = np.unravel_index(s_e, self.state_size_tuple)[0]
			d = np.searchsorted(self.d_discrete, d_index)
		else:
			d = s_e[1]

		if	d <= self.capture_radius:
			print("IN CAPTURE RADIUS")
			self.restart_game() # Restart Game
			return (-self.end_r, self.end_r)
		elif self.t == np.round(self.curtime,2):
			self.restart_game() # Restart Game
			return (self.end_r, -self.end_r)
		else:
			return (-self.step_r, self.step_r)

	def discrete_dynamics(self, a_p, a_e):
		"""
		Takes pursue and evader actions. Returns new position and orientation
		of each agent. Pursuer: (x,y,phi) Evader:(x,y)
		Discrete dynamics integrated by RK4

		param a_p: pursuer action
		param a_e: evader action
		"""
		dt = self.dt
		pos_p = self.p.pos

		k1 = dt*self.p.f_p(pos_p, a_p)
		k2 = dt*self.p.f_p(pos_p+k1/2, a_p)
		k3 = dt*self.p.f_p(pos_p+k2/2, a_p)
		k4 = dt*self.p.f_p(pos_p+k3, a_p)
		pos_p_next = pos_p + (k1+2*k2+2*k3+k4)/6

		pos_e = self.e.pos
		k1 = dt*self.e.f_e(pos_e, a_e)
		k2 = dt*self.e.f_e(pos_e+k1/2, a_e)
		k3 = dt*self.e.f_e(pos_e+k2/2, a_e)
		k4 = dt*self.e.f_e(pos_e+k3, a_e)
		pos_e_next = pos_e + (k1+2*k2+2*k3+k4)/6

		return (pos_p_next, pos_e_next)

	def get_states(self, pos_p, pos_e, discrete_state=True):
		"""
		Returns state of pursuer (phi, dphi) and evader (phi, d) based on
		positions and orientations found in discrete dynamics.

		param pos_p: position and orientation of puruser
		param pos_e: position and orientation of evader
		"""
		phi = np.arctan((pos_e[1] - pos_p[1])/(pos_e[0] - pos_p[0])) - pos_p[2]
		dphi = self.p.w/self.p.R_p*np.clip(pos_p[2], self.p.a_min, self.p.a_max)
		d = np.linalg.norm(pos_p[:2] - pos_e)

		if discrete_state:
			phi_d = np.searchsorted(self.phi_discrete, phi)
			phi_dot_d = np.searchsorted(self.phi_dot_discrete, dphi)
			d_d = np.searchsorted(self.d_discrete, d)

			s_d = np.ravel_multi_index((d_d, phi_d, phi_dot_d), self.state_size_tuple)

			return (s_d, s_d)
		else:
			return np.array([(phi, dphi), (phi, d)])

	def from_discrete_p_action(self, a_p):
		"""
		Returns the continuous action based of the discrete action index
		so the dynamics can be propogated forward.

		param a_p: discrete action of the pursuer
		"""
		a_p_c = self.a_p_discrete[a_p]

		return a_p_c

	def from_discrete_e_action(self, a_e):
		"""
		Returns the continuous action based of the discrete action index
		so the dynamics can be propogated forward.

		param a_e: discrete action of the evader
		"""
		a_e_c = self.a_e_discrete[a_e]

		return a_e_c

	def simulate(self, a_p, a_e, discrete_p_action=False, discrete_e_action=False):
		"""
		Takes pursue and evader actions. Returns next state of game
		Discrete dynamics integrated by RK4

		pa			print(pos_p, pos_e)ram a_p: pursuer action
		param a_e: evader action
		"""
		# Calculate next x and y for pursuer and evader
		if discrete_p_action:
			a_p = self.from_discrete_p_action(a_p)

		if discrete_e_action:
			a_e = self.from_discrete_e_action(a_e)

		pos_p, pos_e = self.discrete_dynamics(a_p, a_e)
		# Check that both agents are inbounds of the map
		# pos_p, pos_e = self.inbounds(pos_p, pos_e)
		self.p.pos = pos_p
		self.e.pos = pos_e
		# Find and update pursuer and evader states
		s_p, s_e = self.get_states(pos_p, pos_e)
		self.p.update_state(s_p)
		self.e.update_state(s_e)
		# Increment Game
		self.increment_game(pos_p, pos_e)
		# Find reward at current positions and time
		r_p, r_e = self.get_reward(s_p, s_e, discrete_state=True)
		# Return reward and new state
		return np.array([(s_p, r_p),(s_e, r_e)])
