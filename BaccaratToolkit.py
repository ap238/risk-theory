import numpy as np
from collections import Counter
import itertools

class BaccaratMarkovChain:
    """
    Markov chain analysis of shoe outcomes (banker/player/tie).
    Supports empirical transition fitting, simulation, and stationary distribution.
    """
    def __init__(self, transition_matrix=None, history=None):
        self.states = ['banker', 'player', 'tie']
        if transition_matrix is not None:
            self.P = np.array([[transition_matrix[s1][s2] for s2 in self.states] for s1 in self.states])
        elif history is not None:
            self.P = self.fit_transition_matrix(history)
        else:
            self.P = np.ones((3,3)) / 3
        self.state_idx = {s: i for i, s in enumerate(self.states)}

    def fit_transition_matrix(self, history):
        """Fit empirical transition matrix from a sequence of outcomes."""
        counts = np.zeros((3,3))
        for (a, b) in zip(history[:-1], history[1:]):
            i, j = self.state_idx[a], self.state_idx[b]
            counts[i, j] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.nan_to_num(counts / row_sums, nan=1/3)
        return P

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

class BaccaratCombinatorics:
    """
    Combinatorial enumeration for banker/player/tie bets.
    Uses Monte Carlo for tractability, but can do exact for small decks.
    """
    @staticmethod
    def enumerate_outcomes(deck=None, num_samples=100000):
        # If deck is None, use 8 decks
        if deck is None:
            deck = [v for v in range(1,14)] * 8 * 4
        outcomes = {'banker': 0, 'player': 0, 'tie': 0}
        for _ in range(num_samples):
            d = deck.copy()
            np.random.shuffle(d)
            # Deal 2 cards each
            player = [d.pop(), d.pop()]
            banker = [d.pop(), d.pop()]
            # Baccarat hand value: sum mod 10
            p_val = sum(player) % 10
            b_val = sum(banker) % 10
            if p_val > b_val:
                outcomes['player'] += 1
            elif b_val > p_val:
                outcomes['banker'] += 1
            else:
                outcomes['tie'] += 1
        total = sum(outcomes.values())
        return {k: v/total for k, v in outcomes.items()}

    @staticmethod
    def bet_probability(bet_type, deck=None, num_samples=100000):
        probs = BaccaratCombinatorics.enumerate_outcomes(deck, num_samples)
        return probs.get(bet_type, 0.0)

class BaccaratEdgeSorting:
    """
    Edge sorting and card counting utilities for baccarat.
    Tracks running count, running edge, and can estimate player/banker advantage.
    """
    def __init__(self, deck=None):
        self.deck = Counter(deck) if deck else Counter({v:8*4 for v in range(1,14)})  # 8 decks
        self.total_cards = sum(self.deck.values())
        self.history = []

    def update(self, card):
        if self.deck[card] > 0:
            self.deck[card] -= 1
            self.total_cards -= 1
            self.history.append(card)

    def running_count(self):
        # High cards (8,9) favor player, low cards (1-7) favor banker
        count = 0
        for card, n in self.deck.items():
            if card in [8,9]:
                count += n
            elif card in range(1,8):
                count -= n
        return count

    def running_edge(self):
        # Estimate edge as difference in high/low card proportion
        high = sum(self.deck[c] for c in [8,9])
        low = sum(self.deck[c] for c in range(1,8))
        if self.total_cards == 0:
            return 0.0
        return (high - low) / self.total_cards

    def get_deck(self):
        return self.deck.copy()

# Example usage:
if __name__ == "__main__":
    # (A) Markov chain
    print("--- Markov Chain ---")
    mc = BaccaratMarkovChain()
    sim = mc.simulate('banker', 50)
    print("Simulated shoe outcomes:", sim)
    print("Stationary distribution:", mc.stationary_distribution())
    # Empirical fit
    mc_emp = BaccaratMarkovChain(history=sim)
    print("Empirical transition matrix:\n", mc_emp.P)

    # (B) Combinatorics
    print("\n--- Combinatorics ---")
    probs = BaccaratCombinatorics.enumerate_outcomes(num_samples=10000)
    print("Monte Carlo bet probabilities:", probs)
    print("Probability of tie:", BaccaratCombinatorics.bet_probability('tie', num_samples=10000))

    # (C) Edge sorting/card counting
    print("\n--- Edge Sorting & Card Counting ---")
    es = BaccaratEdgeSorting()
    for card in [8, 9, 2, 3, 8, 7, 9, 1, 5, 8]:
        es.update(card)
        print(f"After card {card}, running count:", es.running_count(), \
              ", running edge:", es.running_edge())
    print("Current deck:", es.get_deck()) 