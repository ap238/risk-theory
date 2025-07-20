import numpy as np

class SPRT:
    def __init__(self, p0, p1, alpha=0.05, beta=0.05):
        """
        p0: Null hypothesis win probability (e.g., fair deck, 0.5)
        p1: Alternative hypothesis win probability (e.g., player edge, >0.5)
        alpha: Type I error (false positive rate)
        beta: Type II error (false negative rate)
        """
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta
        self.A = beta / (1 - alpha)
        self.B = (1 - beta) / alpha
        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        self.log_likelihood = 0.0
        self.history = []

    def update(self, outcome):
        """
        outcome: 1 for win, 0 for loss (or -1 for loss, 0 for push, 1 for win if desired)
        Updates the log-likelihood ratio and returns a decision:
        'bet_more', 'leave', or 'keep_playing'
        """
        # For blackjack, treat push as 0, win as 1, loss as 0 (or -1 for more granularity)
        # We'll use 1 for win, 0 for loss/push for simplicity
        if outcome == 1:
            llr = np.log(self.p1 / self.p0)
        else:
            llr = np.log((1 - self.p1) / (1 - self.p0))
        self.log_likelihood += llr
        self.history.append(self.log_likelihood)
        if self.log_likelihood <= self.log_A:
            return 'leave'
        elif self.log_likelihood >= self.log_B:
            return 'bet_more'
        else:
            return 'keep_playing'

    def reset(self):
        self.log_likelihood = 0.0
        self.history = []

# Example usage:
if __name__ == "__main__":
    # Suppose p0 = 0.5 (fair), p1 = 0.55 (edge), alpha = beta = 0.05
    sprt = SPRT(p0=0.5, p1=0.55, alpha=0.05, beta=0.05)
    outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    for i, outcome in enumerate(outcomes):
        decision = sprt.update(outcome)
        print(f"Hand {i+1}: outcome={outcome}, log_likelihood={sprt.log_likelihood:.3f}, decision={decision}")
        if decision != 'keep_playing':
            print(f"SPRT decision: {decision.upper()} at hand {i+1}")
            break 