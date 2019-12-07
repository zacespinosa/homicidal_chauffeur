import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader
import matplotlib.pyplot as plt

def random_evader():
	num_d_states = 25
	num_phi_states = 20
	num_phi_d_states = 20
	num_actions = 100
	num_states = num_d_states*num_phi_states*num_phi_d_states

	num_epochs = 1000
	capture_times = np.zeros(num_epochs)

	random_initialization = False

	p = Pursuer()
	e = Evader(num_d_states, num_phi_states, num_phi_d_states, num_actions, np.array([10,10]), learning='Q-learning', load_q=random_initialization)
	s = Simulator(p, e, num_d_states, num_phi_states, num_phi_d_states, num_actions, verbose=True)


	while s.restarts < num_epochs:
		# execute optimal pursuer strategy while training evader
		a_p = p.optimal_strategy(e.pos, p.pos)
		a_e = e.random_strategy()
		p_info, e_info = s.simulate(a_p, a_e, discrete_p_action=False, discrete_e_action=True)
		if s.end_game: s.restart_game()
		capture_times[s.restarts-1] = s.last_capture_time

	print("Evader captured: ", s.num_captures, "/", s.restarts, " times.")

	avg_time = np.average(capture_times)
	print("Average survival time of ", avg_time, "for", num_epochs, "games")
	# plot time till capture for every epoch
	plt.plot(np.arange(num_epochs), capture_times, 'k')
	plt.show()

random_evader()