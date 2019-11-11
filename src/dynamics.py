import numpy as np
"""
Continuous dynamics of the pursuer
"""
def f_p(s, a, w_p, R_p, a_min, a_max):
	s_dot = np.zeros(3)
	s_dot[0] = w_p*np.cos(s[2])
	s_dot[1] = w_p*np.sin(s[2])
	s_dot[2] = w_p/R_p*np.clip(a, a_min, a_max)

	return s_dot

"""
Continuous dynamics of the evader
"""
def f_e(s, a, w_e):
	s_dot = np.zeros(2)
	s_dot[0] = w_e*np.cos(a)
	s_dot[1] = w_e*np.sin(a)

	return s_dot

"""
Discrete dynamics integrated by RK4
"""
def discrete_dynamics(s_p, s_e, a_p, a_e, dt=0.01):
	w_p = 2
	w_e = 1
	a_min = -0.5
	a_max = 0.5
	L = 0.3
	R_p = L/np.tan(a_max)

	k1 = dt*f_p(s_p, a_p, w_p, R_p, a_min, a_max)
	k2 = dt*f_p(s_p+k1/2, a_p, w_p, R_p, a_min, a_max)
	k3 = dt*f_p(s_p+k2/2, a_p, w_p, R_p, a_min, a_max)
	k4 = dt*f_p(s_p+k3, a_p, w_p, R_p, a_min, a_max)
	s_p_next = s_p + (k1+2*k2+2*k3+k4)/6

	k1 = dt*f_e(s_e, a_e, w_e)
	k2 = dt*f_e(s_e+k1/2, a_e, w_e)
	k3 = dt*f_e(s_e+k2/2, a_e, w_e)
	k4 = dt*f_e(s_e+k3, a_e, w_e)
	s_e_next = s_e + (k1+2*k2+2*k3+k4)/6

	return (s_p_next, s_e_next)

s_p = np.zeros(3)
s_e = np.zeros(2)
for i in range(1000):
	(s_p, s_e) = discrete_dynamics(s_p, s_e, 0.1, -0.1)

print("final pursuer state ", s_p)
print("final evader state ", s_e)
