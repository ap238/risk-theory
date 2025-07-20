import numpy as np
import random

class BlackjackHMM:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.N = len(states)
        self.M = len(observations)
        self.start_prob = np.array([start_prob[s] for s in states])
        self.trans_prob = np.array([[trans_prob[s1][s2] for s2 in states] for s1 in states])
        self.emit_prob = np.array([[emit_prob[s][o] for o in observations] for s in states])

    def forward(self, obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.start_prob * self.emit_prob[:, self.observations.index(obs_seq[0])]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.trans_prob[:, j]) * self.emit_prob[j, self.observations.index(obs_seq[t])]
        return alpha

    def backward(self, obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1
        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.trans_prob[i, :] * self.emit_prob[:, self.observations.index(obs_seq[t+1])] * beta[t+1])
        return beta

    def viterbi(self, obs_seq):
        T = len(obs_seq)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = self.start_prob * self.emit_prob[:, self.observations.index(obs_seq[0])]
        for t in range(1, T):
            for j in range(self.N):
                seq_probs = delta[t-1] * self.trans_prob[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) * self.emit_prob[j, self.observations.index(obs_seq[t])]
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        for t in reversed(range(1, T)):
            path[t-1] = psi[t, path[t]]
        state_path = [self.states[i] for i in path]
        return state_path, delta

    def baum_welch(self, obs_seq, n_iter=10):
        T = len(obs_seq)
        obs_idx = [self.observations.index(o) for o in obs_seq]
        for n in range(n_iter):
            alpha = self.forward(obs_seq)
            beta = self.backward(obs_seq)
            xi = np.zeros((T-1, self.N, self.N))
            gamma = np.zeros((T, self.N))
            for t in range(T-1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.N):
                    numer = alpha[t, i] * self.trans_prob[i, :] * self.emit_prob[:, obs_idx[t+1]] * beta[t+1]
                    xi[t, i, :] = numer / (denom + 1e-12)
            gamma = np.sum(xi, axis=2)
            # Update start_prob
            self.start_prob = gamma[0] / np.sum(gamma[0])
            # Update trans_prob
            self.trans_prob = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0)[:, None] + 1e-12)
            # Update emit_prob
            for k in range(self.M):
                mask = np.array(obs_idx) == k
                self.emit_prob[:, k] = np.sum(gamma[mask], axis=0) / (np.sum(gamma, axis=0) + 1e-12)
        return self

    def state_posteriors(self, obs_seq):
        alpha = self.forward(obs_seq)
        beta = self.backward(obs_seq)
        post = alpha * beta
        post /= np.sum(post, axis=1, keepdims=True)
        return post

    def log_likelihood(self, obs_seq):
        alpha = self.forward(obs_seq)
        return np.log(np.sum(alpha[-1]))

def simulate_blackjack_sequence(states, observations, start_prob, trans_prob, emit_prob, length=30, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    state_seq = []
    obs_seq = []
    current_state = np.random.choice(states, p=[start_prob[s] for s in states])
    for _ in range(length):
        state_seq.append(current_state)
        obs = np.random.choice(observations, p=[emit_prob[current_state][o] for o in observations])
        obs_seq.append(obs)
        current_state = np.random.choice(states, p=[trans_prob[current_state][s] for s in states])
    return state_seq, obs_seq

if __name__ == "__main__":
    # Define states and observations
    states = ['FairDeck', 'HotDeck', 'ColdDeck']
    observations = ['Win', 'Lose', 'Draw']
    start_prob = {'FairDeck': 0.8, 'HotDeck': 0.1, 'ColdDeck': 0.1}
    trans_prob = {
        'FairDeck': {'FairDeck': 0.90, 'HotDeck': 0.05, 'ColdDeck': 0.05},
        'HotDeck': {'FairDeck': 0.10, 'HotDeck': 0.85, 'ColdDeck': 0.05},
        'ColdDeck': {'FairDeck': 0.10, 'HotDeck': 0.05, 'ColdDeck': 0.85},
    }
    emit_prob = {
        'FairDeck': {'Win': 0.44, 'Lose': 0.48, 'Draw': 0.08},
        'HotDeck': {'Win': 0.55, 'Lose': 0.38, 'Draw': 0.07},
        'ColdDeck': {'Win': 0.35, 'Lose': 0.60, 'Draw': 0.05},
    }
    # Simulate a blackjack sequence
    true_states, obs_seq = simulate_blackjack_sequence(states, observations, start_prob, trans_prob, emit_prob, length=30, seed=21)
    print("Observed sequence:", obs_seq)
    print("True hidden states:", true_states)
    hmm = BlackjackHMM(states, observations, start_prob, trans_prob, emit_prob)
    viterbi_path, _ = hmm.viterbi(obs_seq)
    print("Viterbi inferred states:", viterbi_path)
    print("Forward probabilities (alpha):\n", hmm.forward(obs_seq))
    print("Backward probabilities (beta):\n", hmm.backward(obs_seq))
    print("State posteriors (P(state|obs)):\n", hmm.state_posteriors(obs_seq))
    print("Log-likelihood:", hmm.log_likelihood(obs_seq))
    # Unsupervised learning (Baum-Welch)
    hmm.baum_welch(obs_seq, n_iter=5)
    print("Learned transition probabilities:\n", hmm.trans_prob)
    print("Learned emission probabilities:\n", hmm.emit_prob) 