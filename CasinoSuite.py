from Blackjack_HMM import HiddenMarkovModel as BlackjackHMM
from BaccaratToolkit import BaccaratMarkovChain, BaccaratCombinatorics, BaccaratEdgeSorting
from PokerToolkit import BayesianHandRange, NashEquilibriumCFR, PokerDynamicProgramming, PokerInformationTheory, PokerReinforcementLearning
from DiceGamesToolkit import CrapsMarkovChain, SicBoOdds, YahtzeeDP, CeeLoOdds
from SlotMachine_HMM import SlotMachineHMM
from Roulette_HMM import RouletteHMM

class CasinoSuite:
    """
    Unified casino suite for all major games: Blackjack, Baccarat, Poker, Dice Games, Slot Machine, Roulette.
    Provides simulation, odds calculation, and Monte Carlo analysis for each game.
    """
    def __init__(self):
        # Initialize all toolkits
        self.blackjack = BlackjackHMM
        self.baccarat = {
            'markov': BaccaratMarkovChain(),
            'combinatorics': BaccaratCombinatorics(),
            'edge_sorting': BaccaratEdgeSorting()
        }
        self.poker = {
            'bayesian': BayesianHandRange,
            'cfr': NashEquilibriumCFR,
            'dp': PokerDynamicProgramming,
            'info': PokerInformationTheory,
            'rl': PokerReinforcementLearning
        }
        self.dice = {
            'craps': CrapsMarkovChain(),
            'sicbo': SicBoOdds(),
            'yahtzee': YahtzeeDP(),
            'ceelo': CeeLoOdds()
        }
        self.slot = SlotMachineHMM
        self.roulette = RouletteHMM

    def list_games(self):
        return ['blackjack', 'baccarat', 'poker', 'dice', 'slot', 'roulette']

    def run_blackjack(self):
        print("--- Blackjack ---")
        # Example: run HMM on a sample sequence
        states = ['Fair', 'Loaded']
        observations = ['1', '2', '3', '4', '5', '6']
        start_prob = {'Fair': 0.5, 'Loaded': 0.5}
        trans_prob = {
            'Fair': {'Fair': 0.95, 'Loaded': 0.05},
            'Loaded': {'Fair': 0.1, 'Loaded': 0.9}
        }
        emit_prob = {
            'Fair': {o: 1/6 for o in observations},
            'Loaded': {**{str(i): 0.1 for i in range(1,6)}, '6': 0.5}
        }
        hmm = self.blackjack(states, observations, start_prob, trans_prob, emit_prob)
        obs_seq = ['3', '1', '6', '6', '2', '4', '6', '6', '5', '6']
        print("Observation sequence:", obs_seq)
        print("Viterbi best state path:", hmm.viterbi(obs_seq)[0])
        print("Forward probabilities (alpha):\n", hmm.forward(obs_seq))
        print("Backward probabilities (beta):\n", hmm.backward(obs_seq))

    def run_baccarat(self):
        print("--- Baccarat ---")
        mc = self.baccarat['markov']
        sim = mc.simulate('banker', 20)
        print("Simulated shoe outcomes:", sim)
        print("Stationary distribution:", mc.stationary_distribution())
        print("Standard bet probabilities:", self.baccarat['combinatorics'].enumerate_outcomes(num_samples=10000))
        es = self.baccarat['edge_sorting']
        for card in [8, 9, 2, 3, 8, 7]:
            es.update(card)
            print(f"After card {card}, running count:", es.running_count(), ", running edge:", es.running_edge())

    def run_poker(self):
        print("--- Poker ---")
        all_hands = [('Ah','Kh'), ('2c','2d'), ('Qs','Jd')]
        bayes = self.poker['bayesian'](all_hands)
        bayes.update(board=['As','Kd','2h'], actions=['bet','call'])
        print("Posterior hand range:", bayes.get_range())
        cfr = self.poker['cfr'](game='holdem')
        cfr.train(iterations=100)
        print("Strategy for state:", cfr.get_strategy(state=None))
        dp = self.poker['dp']((100,100), 10, 0.7)
        dp.solve()
        prior = {'AhKh': 0.5, '2c2d': 0.5}
        posterior = {'AhKh': 0.8, '2c2d': 0.2}
        print("Hand entropy:", self.poker['info'].hand_entropy(prior))
        print("Mutual information:", self.poker['info'].mutual_information(prior, posterior))
        rl = self.poker['rl'](state_space=['s1','s2'], action_space=['fold','call','raise'])
        rl.train(episodes=100)
        print("Policy for state s1:", rl.get_policy('s1'))

    def run_dice(self):
        print("--- Dice Games ---")
        craps = self.dice['craps']
        win, lose = craps.pass_line_odds()
        print(f"Craps pass line win: {win:.4f}, lose: {lose:.4f}")
        sim = craps.simulate(10000)
        print(f"Craps Monte Carlo: win={sim['win_prob']:.4f}, lose={sim['lose_prob']:.4f}, mean={sim['mean']:.4f}, variance={sim['variance']:.4f}")
        print("\nSic Bo odds:")
        for k, v in self.dice['sicbo'].odds_display().items():
            print(f"{k}: {v:.4f}")
        sim = self.dice['sicbo'].monte_carlo_sim('sum', 10, n=10000)
        print(f"Sic Bo Monte Carlo for sum 10: prob={sim['prob']:.4f}, mean={sim['mean']:.4f}, variance={sim['variance']:.4f}")
        yahtzee = self.dice['yahtzee']
        dice = (1,2,3,4,5)
        exp_val = yahtzee.expected_value(dice, 2, set())
        print(f"Yahtzee expected value for dice {dice} with 2 rolls left: {exp_val:.2f}")
        sim = yahtzee.monte_carlo_sim(n=1000)
        print(f"Yahtzee Monte Carlo: mean={sim['mean']:.2f}, variance={sim['variance']:.2f}")
        ceelo = self.dice['ceelo']
        print("Cee-lo odds:")
        for k, v in ceelo.odds_display().items():
            print(f"{k}: {v:.4f}")
        sim = ceelo.monte_carlo_sim(n=10000)
        print("Cee-lo Monte Carlo distribution:")
        for k, v in sim.items():
            print(f"{k}: {v:.4f}")

    def run_slot(self):
        print("--- Slot Machine ---")
        # Example: run slot HMM on a simulated sequence
        states = ['Fair', 'Loose', 'Tight', 'Broken']
        observations = ['Jackpot', 'SmallWin', 'Loss', 'Error']
        start_prob = {'Fair': 0.7, 'Loose': 0.1, 'Tight': 0.15, 'Broken': 0.05}
        trans_prob = {
            'Fair':   {'Fair': 0.85, 'Loose': 0.05, 'Tight': 0.08, 'Broken': 0.02},
            'Loose':  {'Fair': 0.10, 'Loose': 0.80, 'Tight': 0.05, 'Broken': 0.05},
            'Tight':  {'Fair': 0.10, 'Loose': 0.05, 'Tight': 0.80, 'Broken': 0.05},
            'Broken': {'Fair': 0.20, 'Loose': 0.10, 'Tight': 0.10, 'Broken': 0.60},
        }
        emit_prob = {
            'Fair':   {'Jackpot': 0.01, 'SmallWin': 0.15, 'Loss': 0.83, 'Error': 0.01},
            'Loose':  {'Jackpot': 0.03, 'SmallWin': 0.30, 'Loss': 0.65, 'Error': 0.02},
            'Tight':  {'Jackpot': 0.005, 'SmallWin': 0.10, 'Loss': 0.88, 'Error': 0.015},
            'Broken': {'Jackpot': 0.0, 'SmallWin': 0.0, 'Loss': 0.5, 'Error': 0.5},
        }
        hmm = self.slot(states, observations, start_prob, trans_prob, emit_prob)
        obs_seq = ['Loss', 'SmallWin', 'Loss', 'Jackpot', 'Loss', 'Loss', 'Error', 'Loss', 'SmallWin', 'Loss']
        print("Observation sequence:", obs_seq)
        print("Viterbi best state path:", hmm.viterbi(obs_seq)[0])
        print("Forward probabilities (alpha):\n", hmm.forward(obs_seq))
        print("Backward probabilities (beta):\n", hmm.backward(obs_seq))

    def run_roulette(self):
        print("--- Roulette ---")
        # Example: run roulette HMM on a simulated sequence
        states = ['Fair', 'BiasedRed', 'BiasedBlack']
        observations = ['Red', 'Black', 'Green']
        start_prob = {'Fair': 0.8, 'BiasedRed': 0.1, 'BiasedBlack': 0.1}
        trans_prob = {
            'Fair': {'Fair': 0.90, 'BiasedRed': 0.05, 'BiasedBlack': 0.05},
            'BiasedRed': {'Fair': 0.10, 'BiasedRed': 0.85, 'BiasedBlack': 0.05},
            'BiasedBlack': {'Fair': 0.10, 'BiasedRed': 0.05, 'BiasedBlack': 0.85},
        }
        emit_prob = {
            'Fair': {'Red': 18/37, 'Black': 18/37, 'Green': 1/37},
            'BiasedRed': {'Red': 0.7, 'Black': 0.25, 'Green': 0.05},
            'BiasedBlack': {'Red': 0.25, 'Black': 0.7, 'Green': 0.05},
        }
        hmm = self.roulette(states, observations, start_prob, trans_prob, emit_prob)
        obs_seq = ['Red', 'Black', 'Red', 'Red', 'Green', 'Black', 'Red', 'Red', 'Black', 'Red']
        print("Observation sequence:", obs_seq)
        print("Viterbi best state path:", hmm.viterbi(obs_seq)[0])
        print("Forward probabilities (alpha):\n", hmm.forward(obs_seq))
        print("Backward probabilities (beta):\n", hmm.backward(obs_seq))

if __name__ == "__main__":
    suite = CasinoSuite()
    print("Welcome to the CasinoSuite! Available games:")
    for i, g in enumerate(suite.list_games()):
        print(f"{i+1}. {g}")
    # Example: run all games
    suite.run_blackjack()
    suite.run_baccarat()
    suite.run_poker()
    suite.run_dice()
    suite.run_slot()
    suite.run_roulette() 