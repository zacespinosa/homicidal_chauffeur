import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader

def model(): 
    pass

def test(): 
	p = Pursuer()
	e = Evader(np.array([10,10]))
	s = Simulator(p, e)
	for i in range(int(600/.01)):
		a_e = random.random()*2*np.pi
		a_p = random.uniform(p.a_min, p.a_max) 
		p_info, e_info = s.simulate(a_p, a_e)

test()