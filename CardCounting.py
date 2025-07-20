import numpy as np

class CardCounting:
    def __init__(self, running_count=0, decks=6):
        self.running_count = running_count
        self.decks = decks
        self.cards_remaining = 52 * decks

    def update_count(self, card):
        # Hi-Lo system: 2-6 = +1, 7-9 = 0, 10-A = -1
        if card in [2,3,4,5,6]:
            self.running_count += 1
        elif card in [10, 'J', 'Q', 'K', 'A']:
            self.running_count -= 1
        # 7,8,9 are zero
        self.cards_remaining -= 1

    def true_count(self):
        decks_remaining = self.cards_remaining / 52
        if decks_remaining == 0:
            return 0
        return self.running_count / decks_remaining

    @staticmethod
    def bayesian_update(deck_counts, observed_card):
        # Simple Bayesian update: remove observed card from deck
        new_deck = deck_counts.copy()
        if new_deck[observed_card] > 0:
            new_deck[observed_card] -= 1
        return new_deck

    @staticmethod
    def kelly_criterion(edge, bankroll, odds=1):
        # edge: expected win probability - loss probability
        # odds: payout odds (default 1:1)
        if edge <= 0:
            return 0
        return bankroll * edge / odds 