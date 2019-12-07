import random as random
import numpy as np
from dynamics import Simulator, Pursuer, Evader
import matplotlib.pyplot as plt

def train_evader():
    num_d_states = 25
    num_phi_states = 20
    num_phi_d_states = 20
    num_actions = 100
    num_states = num_d_states*num_phi_states*num_phi_d_states

    num_epochs = 20
    capture_times = np.zeros(num_epochs)
    loss = [] 
    mae = [] 

    random_initialization = False

    p = Pursuer()
    e = Evader(num_d_states, num_phi_states, num_phi_d_states, num_actions, np.array([10,10]), learning='dqn')
    s = Simulator(p, e, num_d_states, num_phi_states, num_phi_d_states, num_actions, verbose=False)

    while s.restarts < num_epochs:
        # execute optimal pursuer strategy while training evader
        a_p = p.optimal_strategy(e.pos, p.pos)
        # execute DQN policy for evader
        s_e = e.s
        a_e = e.dqn_strategy(s_e, s.restarts)
        p_info, e_info = s.simulate(a_p, a_e, discrete_p_action=False, discrete_e_action=True)

        s_e_next, r_e = e_info
        metrics = e.update_ddqn_strategy(s_e, s_e_next, a_e, r_e, e.end_game)
        if metrics["updated"]: 
            loss.append(metrics["loss"])
            mae.append(metrics["mae"])

        if s.end_game: 
            print("Starting Game: ", s.restarts, "/", num_epochs)
            s.restart_game()

        capture_times[s.restarts-1] = s.last_capture_time

    # save model
    model_json = e.qnetwork.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    e.qnetwork.save_weights("model.h5")
    print("Saved model to disk")

    print("Evader captured: ", s.num_captures, "/", s.restarts, " times.")
    # Save network

    # plot time till capture for every epoch
    plt.plot(np.arange(num_epochs), capture_times, 'k')
    plt.title('Capture Time')
    plt.ylabel('Time')
    plt.xlabel('Game')
    plt.show()

    plt.plot(np.arange(len(loss)), loss, 'k')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

    plt.plot(np.arange(len(mae)), mae, 'k')
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Iteration')
    plt.show()

train_evader()