from collections import defaultdict

class DynamicProgramming:
    def __init__(self, deck=None):
        # deck: dict of card -> count (optional, for card counting)
        self.deck = deck
        self.actions = ['hit', 'stand']  # Can be extended to 'double', 'split', etc.
        self.states = []  # Will be filled with (player_total, dealer_upcard, is_soft)
        self.policy = {}
        self.value = {}

    def enumerate_states(self, min_total=4, max_total=21, dealer_cards=range(2,12)):
        # is_soft: True if hand contains an ace counted as 11
        for player_total in range(min_total, max_total+1):
            for dealer_upcard in dealer_cards:
                for is_soft in [False, True]:
                    self.states.append((player_total, dealer_upcard, is_soft))

    def reward(self, player_total, dealer_total):
        # Returns +1 for win, -1 for loss, 0 for push
        if player_total > 21:
            return -1
        if dealer_total > 21:
            return 1
        if player_total > dealer_total:
            return 1
        if player_total < dealer_total:
            return -1
        return 0

    def value_iteration(self, gamma=1.0, theta=1e-4, max_iter=1000):
        self.enumerate_states()
        V = defaultdict(float)
        policy = {}
        for _ in range(max_iter):
            delta = 0
            for state in self.states:
                player_total, dealer_upcard, is_soft = state
                action_values = {}
                for action in self.actions:
                    if action == 'stand':
                        # Dealer plays out hand (simplified: dealer hits until 17)
                        dealer_total = dealer_upcard
                        while dealer_total < 17:
                            dealer_total += 6  # Approximate average card value
                        action_values[action] = self.reward(player_total, dealer_total)
                    elif action == 'hit':
                        # Simplified: assume next card is 10 (most common)
                        next_total = player_total + 10
                        next_is_soft = is_soft
                        if is_soft and next_total > 21:
                            next_total -= 10  # Convert ace from 11 to 1
                            next_is_soft = False
                        if next_total > 21:
                            action_values[action] = -1
                        else:
                            action_values[action] = V[(next_total, dealer_upcard, next_is_soft)]
                if action_values:
                    best_action = max(action_values, key=lambda k: action_values[k])
                    delta = max(delta, abs(V[state] - action_values[best_action]))
                    V[state] = action_values[best_action]
                    policy[state] = best_action
            if delta < theta:
                break
        self.value = dict(V)
        self.policy = policy
        return policy, V

    def get_action(self, player_total, dealer_upcard, is_soft):
        return self.policy.get((player_total, dealer_upcard, is_soft), 'stand') 