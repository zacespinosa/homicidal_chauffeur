import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader
import matplotlib.pyplot as plt

def test_dqn_evader():
    num_d_states = 25
    num_phi_states = 20
    num_phi_d_states = 20
    num_actions = 100
    num_states = num_d_states*num_phi_states*num_phi_d_states

    num_epochs = 20
    capture_times = np.zeros(num_epochs)
    loss = [] 
    mae = [] 

    p = Pursuer()
    e = Evader(num_d_states, num_phi_states, num_phi_d_states, num_actions, np.array([10,10]), learning='dqn', load_q=True)
    s = Simulator(p, e, num_d_states, num_phi_states, num_phi_d_states, num_actions, verbose=False)

    while s.restarts < num_epochs:
        # execute optimal pursuer strategy while training evader
        a_p = p.optimal_strategy(e.pos, p.pos)
        # execute DQN policy for evader
        s_e = e.s
        a_e = e.dqn_strategy(s_e, s.restarts)
        if a_e != 0: print(a_e)
        p_info, e_info = s.simulate(a_p, a_e, discrete_p_action=False, discrete_e_action=True)

        s_e_next, r_e = e_info

        if s.end_game: 
            print("Starting Game: ", s.restarts, "/", num_epochs)
            s.restart_game()

        capture_times[s.restarts-1] = s.last_capture_time

    print("Evader captured: ", s.num_captures, "/", s.restarts, " times.")
    avg_time = np.average(capture_times)
    print("Average survival time of ", avg_time, "for", num_epochs, "games")

    # plot time till capture for every epoch
    plt.plot(np.arange(num_epochs), capture_times, 'k')
    plt.title('Capture Time')
    plt.ylabel('Time')
    plt.xlabel('Game')
    plt.show()

test_dqn_evader()