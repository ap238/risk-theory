import numpy as np

class RenewalTheory:
    def __init__(self, shoe_size=6, hands_per_shoe=80, player_edge=0.01):
        self.shoe_size = shoe_size
        self.hands_per_shoe = hands_per_shoe
        self.player_edge = player_edge  # Expected value per hand

    def simulate_shoes(self, num_shoes=1000, bet_size=1):
        """Simulate multiple shoes, return total and per-shoe expected value."""
        shoe_results = []
        for _ in range(num_shoes):
            # Each hand is a Bernoulli trial with win prob = 0.5 + edge/2, lose = 0.5 - edge/2
            wins = np.random.binomial(self.hands_per_shoe, 0.5 + self.player_edge/2)
            losses = self.hands_per_shoe - wins
            profit = (wins - losses) * bet_size
            shoe_results.append(profit)
        return np.array(shoe_results)

    def expected_value_per_shoe(self, bet_size=1):
        """Analytical expected value per shoe."""
        return self.hands_per_shoe * self.player_edge * bet_size

    def session_statistics(self, num_sessions=1000, shoes_per_session=10, bet_size=1):
        """Simulate sessions, return mean, std, and CLT confidence interval for total profit."""
        session_profits = []
        for _ in range(num_sessions):
            profits = self.simulate_shoes(num_shoes=shoes_per_session, bet_size=bet_size)
            session_profits.append(np.sum(profits))
        mean = np.mean(session_profits)
        std = np.std(session_profits)
        # 95% CLT confidence interval
        ci = (mean - 1.96*std/np.sqrt(num_sessions), mean + 1.96*std/np.sqrt(num_sessions))
        return {'mean': mean, 'std': std, '95%_CI': ci, 'all_profits': session_profits} 