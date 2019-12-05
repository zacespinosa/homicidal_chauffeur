import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader

def model():
    pass

def test():
	p = Pursuer()
	e = Evader(np.array([10,10]))
	s = Simulator(p, e, 10, 50, 20, 10)

	s_p = np.zeros(3)
	s_e = np.array([10, 10])

	for i in range(int(600/.01)):
		a_e = e.optimal_strategy(e.pos, p.pos, p.R_p)
		a_p = p.optimal_strategy(e.pos, p.pos)
		p_info, e_info = s.simulate(a_p, a_e)

test()
