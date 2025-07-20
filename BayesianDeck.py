"""
risk-theory: A library of Python programs demonstrating mathematical algorithms for probability.
Module: BayesianDeck
This module demonstrates Bayesian updating for card deck probabilities, as used in card games like Blackjack.
"""
import numpy as np
from collections import Counter

class BayesianDeck:
    """
    BayesianDeck models a deck of cards and demonstrates Bayesian updating of card probabilities
    as cards are drawn. Useful for understanding probability updates in card games.
    """
    def __init__(self, deck_counts=None):
        """
        Initialize the deck. If deck_counts is None, use a standard 52-card deck.
        Args:
            deck_counts (dict or Counter, optional): Mapping of card values to counts.
        """
        # deck_counts: dict or Counter of card -> count
        if deck_counts is None:
            # Standard single deck
            self.deck = Counter({v:4 for v in [2,3,4,5,6,7,8,9,10,'J','Q','K','A']})
        else:
            self.deck = Counter(deck_counts)
        self.total_cards = sum(self.deck.values())
        self.prior = self._get_prob_dist()

    def _get_prob_dist(self):
        """
        Compute the current probability distribution over remaining cards in the deck.
        Returns:
            dict: Card value -> probability
        """
        total = sum(self.deck.values())
        if total == 0:
            return {k: 0.0 for k in self.deck}
        return {k: v/total for k, v in self.deck.items()}

    def update(self, card):
        """
        Update the deck and probability distribution after drawing a card.
        Args:
            card: The card drawn (e.g., 10, 'A', etc.)
        """
        if self.deck[card] > 0:
            self.deck[card] -= 1
            self.total_cards -= 
        self.prior = self._get_prob_dist()

    def expected_player_edge(self):
        """
        Demonstration: Estimate player's edge as probability of drawing 10 or Ace as first card.
        Returns:
            float: Probability of drawing 10 or Ace as first card.
        """
        # Simplified: estimate edge as probability player draws 10/A vs dealer
        # (In reality, edge calculation is more complex and depends on rules/strategy)
        prob_10 = self.prior[10] + self.prior.get('J',0) + self.prior.get('Q',0) + self.prior.get('K',0)
        prob_A = self.prior.get('A',0)
        # Example: edge = prob(player gets blackjack) - prob(dealer gets blackjack)
        # Assume both draw independently
        edge = prob_10 * prob_A - (prob_10 * prob_A)  # This cancels, but you can extend
        # For demonstration, return prob(player draws 10 or A as first card)
        return prob_10 + prob_A

    def get_posterior(self):
        """
        Get the current posterior probability distribution over cards.
        Returns:
            dict: Card value -> probability
        """
        return self.prior.copy()

# Example usage:
if __name__ == "__main__":
    print(__doc__)
    deck = BayesianDeck()
    draws = [10, 'A', 2, 5, 10, 'Q', 7]
    print("Initial prior:", deck.get_posterior())
    for card in draws:
        deck.update(card)
        print(f"After drawing {card}: posterior =", deck.get_posterior(), \
              ", expected player edge =", deck.expected_player_edge()) 