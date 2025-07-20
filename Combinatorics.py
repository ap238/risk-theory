import math
from scipy.stats import hypergeom

class Combinatorics:
    @staticmethod
    def hypergeometric(deck_size, num_success, num_drawn, num_success_drawn):
        """Probability of drawing num_success_drawn successes in num_drawn from deck_size with num_success successes."""
        return hypergeom.pmf(num_success_drawn, deck_size, num_success, num_drawn)

    @staticmethod
    def enumerate_deck(deck):
        """Return a dict of card counts for a given deck list."""
        from collections import Counter
        return dict(Counter(deck))

    @staticmethod
    def outcome_tree(player_hand, dealer_card, deck, max_depth=3):
        """Recursively enumerate all possible outcomes for a given hand (limited by max_depth for tractability)."""
        # This is a stub for a full tree; real implementation would be more complex.
        if max_depth == 0 or sum(deck.values()) == 0:
            return [player_hand]
        outcomes = []
        for card, count in deck.items():
            if count > 0:
                new_deck = deck.copy()
                new_deck[card] -= 1
                outcomes += Combinatorics.outcome_tree(player_hand + [card], dealer_card, new_deck, max_depth-1)
        return outcomes 