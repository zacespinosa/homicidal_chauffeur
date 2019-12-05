import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader

def model():
    pass

def test():
	p = Pursuer()
	e = Evader(np.array([10,10]))
	s = Simulator(p, e)

	s_p = np.zeros(3)
	s_e = np.array([10, 10])

	for i in range(int(600/.01)):
		a_e = e.optimal_strategy(e.pos, p.pos, p.R_p)
		# a_e = e.learned_strategy(s_e)
		a_p = p.optimal_strategy(e.pos, p.pos)

		p_info, e_info = s.simulate(a_p, a_e)

		s_e_next, r_e = e_info
		s_p_next, r_p = p_info

		# e.update_learned_strategy(s_e, s_e_next, a_e, r_e, i, e.end_game)
		if s.end_game: s.restart_game()

		s_p = s_p_next
		s_e = s_e_next

test()
