import numpy as np

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.P = np.array(transition_matrix)
        self.state_idx = {s: i for i, s in enumerate(states)}

    def next_state(self, current_state):
        idx = self.state_idx[current_state]
        return np.random.choice(self.states, p=self.P[idx])

    def simulate(self, start_state, steps):
        state = start_state
        history = [state]
        for _ in range(steps):
            state = self.next_state(state)
            history.append(state)
        return history

    def stationary_distribution(self):
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        stat = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        stat = stat[:, 0]
        return stat / stat.sum() 