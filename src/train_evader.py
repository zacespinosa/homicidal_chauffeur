import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader

def train_evader():
	num_d_states = 10
	num_phi_states = 50
	num_phi_d_states = 20
	num_actions = 10
	num_states = num_d_states*num_phi_states*num_phi_d_states

	p = Pursuer()
	e = Evader(num_d_states, num_phi_states, num_phi_d_states, num_actions, np.array([10,10]))
	s = Simulator(p, e, num_d_states, num_phi_states, num_phi_d_states, num_actions)


	for i in range(10):
		# execute optimal pursuer strategy while training evader
		a_p = p.optimal_strategy(e.pos, p.pos)

		# execute Q Learning policy for evader
		state = e.s
		a_e = e.qLearningPolicy(state)
		p_info, e_info = s.simulate(a_p, a_e, discrete_p_action=False, discrete_e_action=True)

		new_state = e_info[0]
		r_e = e_info[1]

		e.updateQ(new_state, state, a_e, r_e)

train_evader()
