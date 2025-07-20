import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CasinoMathTools:
    """
    Universal and game-specific math/statistics tools for casino games.
    Includes visualization and advanced analytics.
    """
    # --- Universal Tools ---
    @staticmethod
    def confidence_interval(data, alpha=0.05):
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        z = 1.96  # 95% CI
        ci = (mean - z*std/np.sqrt(n), mean + z*std/np.sqrt(n))
        return mean, std, ci

    @staticmethod
    def clt_demo(data, sample_size=100, num_samples=1000):
        means = [np.mean(np.random.choice(data, sample_size)) for _ in range(num_samples)]
        plt.hist(means, bins=30, alpha=0.7)
        plt.title('Central Limit Theorem Demo')
        plt.xlabel('Sample Mean')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def entropy(probs):
        probs = np.array(probs)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # --- Spectral Theorem / Spectral Analysis ---
    @staticmethod
    def spectral_analysis(matrix, show_plot=True, title='Spectrum of Operator (Eigenvalues)'):
        """
        Perform spectral decomposition of a normal operator (matrix).
        Returns eigenvalues, eigenvectors, and (optionally) plots the spectrum.
        """
        eigvals, eigvecs = np.linalg.eig(matrix)
        print("Eigenvalues:", eigvals)
        print("Eigenvectors (columns):\n", eigvecs)
        if show_plot:
            plt.figure(figsize=(6,4))
            plt.scatter(eigvals.real, eigvals.imag, c='b', marker='o')
            plt.axhline(0, color='gray', lw=0.5)
            plt.axvline(0, color='gray', lw=0.5)
            plt.title(title)
            plt.xlabel("Real part")
            plt.ylabel("Imaginary part")
            plt.grid(True)
            plt.show()
        return eigvals, eigvecs

    # --- Blackjack ---
    @staticmethod
    def kelly_criterion(edge, bankroll, odds=1):
        if edge <= 0:
            return 0
        return bankroll * edge / odds

    @staticmethod
    def edge_heatmap(counts, edges):
        plt.figure(figsize=(8,4))
        sns.heatmap(np.array([edges]), annot=True, fmt='.2f', cmap='coolwarm', xticklabels=counts)
        plt.title('Blackjack Edge Heatmap')
        plt.xlabel('Running Count')
        plt.yticks([])
        plt.show()

    # --- Baccarat ---
    @staticmethod
    def markov_chain_heatmap(P, states):
        plt.figure(figsize=(6,5))
        sns.heatmap(P, annot=True, fmt='.2f', cmap='Blues', xticklabels=states, yticklabels=states)
        plt.title('Baccarat Markov Chain Transition Matrix')
        plt.show()

    @staticmethod
    def edge_sorting_plot(counts, edges):
        plt.plot(counts, edges, marker='o')
        plt.title('Baccarat Edge Sorting Over Time')
        plt.xlabel('Card Draw Number')
        plt.ylabel('Edge')
        plt.show()

    # --- Poker ---
    @staticmethod
    def hand_range_heatmap(hand_matrix, xlabels, ylabels):
        plt.figure(figsize=(8,6))
        sns.heatmap(hand_matrix, annot=False, cmap='YlGnBu', xticklabels=xlabels, yticklabels=ylabels)
        plt.title('Poker Hand Range Heatmap')
        plt.xlabel('Opponent Hand')
        plt.ylabel('Player Hand')
        plt.show()

    @staticmethod
    def regret_convergence_plot(regrets):
        plt.plot(regrets)
        plt.title('Nash Equilibrium Regret Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Average Regret')
        plt.show()

    # --- Dice Games ---
    @staticmethod
    def monte_carlo_histogram(results, title='Monte Carlo Results', bins=20):
        plt.hist(results, bins=bins, alpha=0.7)
        plt.title(title)
        plt.xlabel('Outcome')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def expected_value_table(bet_types, evs):
        print("Expected Value Table:")
        for b, e in zip(bet_types, evs):
            print(f"{b}: {e:.4f}")

    # --- Slot/Roulette ---
    @staticmethod
    def bankroll_evolution(bankrolls):
        plt.plot(bankrolls)
        plt.title('Bankroll Evolution Over Time')
        plt.xlabel('Game Number')
        plt.ylabel('Bankroll')
        plt.show()

    @staticmethod
    def entropy_over_time(entropies):
        plt.plot(entropies)
        plt.title('Entropy of State Over Time')
        plt.xlabel('Step')
        plt.ylabel('Entropy (bits)')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Universal
    data = np.random.binomial(1, 0.49, 1000)
    mean, std, ci = CasinoMathTools.confidence_interval(data)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}, 95% CI: {ci}")
    CasinoMathTools.clt_demo(data)
    print("Entropy:", CasinoMathTools.entropy([0.49, 0.51]))

    # Spectral theorem: Markov chain (Baccarat/Craps)
    P = np.array([[0.8,0.1,0.1],[0.2,0.7,0.1],[0.2,0.2,0.6]])
    CasinoMathTools.spectral_analysis(P, title='Baccarat Markov Chain Spectrum')

    # Spectral theorem: Slot/Roulette HMM transition matrix
    slot_P = np.array([[0.85,0.05,0.08,0.02],[0.10,0.80,0.05,0.05],[0.10,0.05,0.80,0.05],[0.20,0.10,0.10,0.60]])
    CasinoMathTools.spectral_analysis(slot_P, title='Slot Machine Markov Spectrum')

    # Spectral theorem: Poker/Baccarat covariance/transition matrix
    cov = np.cov(np.random.randn(100,3).T)
    CasinoMathTools.spectral_analysis(cov, title='Poker/Baccarat Covariance Spectrum')

    # Blackjack
    counts = list(range(-10, 11))
    edges = [0.01 * c for c in counts]
    CasinoMathTools.edge_heatmap(counts, edges)
    print("Kelly bet for edge 0.05, bankroll 1000:", CasinoMathTools.kelly_criterion(0.05, 1000))

    # Baccarat
    states = ['banker', 'player', 'tie']
    CasinoMathTools.markov_chain_heatmap(P, states)
    CasinoMathTools.edge_sorting_plot(list(range(10)), np.random.randn(10))

    # Poker
    hand_matrix = np.random.rand(5,5)
    CasinoMathTools.hand_range_heatmap(hand_matrix, [f'H{i}' for i in range(5)], [f'H{i}' for i in range(5)])
    CasinoMathTools.regret_convergence_plot(np.cumsum(np.random.randn(100)))

    # Dice
    results = np.random.randint(0, 10, 1000)
    CasinoMathTools.monte_carlo_histogram(results, title='Dice Monte Carlo')
    CasinoMathTools.expected_value_table(['bet1','bet2','bet3'], [0.1, -0.05, 0.02])

    # Slot/Roulette
    bankrolls = np.cumsum(np.random.randn(100)) + 1000
    CasinoMathTools.bankroll_evolution(bankrolls)
    entropies = np.linspace(4, 2, 100)
    CasinoMathTools.entropy_over_time(entropies) 