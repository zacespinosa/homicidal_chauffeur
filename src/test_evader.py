import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader

def test_evader():
	num_d_states = 25
	num_phi_states = 20
	num_phi_d_states = 20
	num_actions = 10
	num_states = num_d_states*num_phi_states*num_phi_d_states

	num_epochs = 1000

	p = Pursuer()
	e = Evader(num_d_states, num_phi_states, num_phi_d_states, num_actions, np.array([10,10]), load_q=True)
	s = Simulator(p, e, num_d_states, num_phi_states, num_phi_d_states, num_actions, verbose=True)


	while s.restarts < 1:
		# execute optimal pursuer strategy while training evader
		a_p = p.optimal_strategy(e.pos, p.pos)
		# a_e = e.optimal_strategy(e.pos, p.pos, p.R_p)

		# execute Q Learning policy for evader
		state = e.s
		a_e = e.qLearningPolicy(state)
		p_info, e_info = s.simulate(a_p, a_e, discrete_p_action=False, discrete_e_action=True)

		new_state = e_info[0]
		r_e = e_info[1]

		# e.updateQ(new_state, state, a_e, r_e)

	# print("Evader captured: ", s.num_captures, "/", s.restarts, " times.")

test_evader()
