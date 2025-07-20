import numpy as np

class MonteCarloSimulation:
    def __init__(self, deck=None, policy=None, bet_size=1, bankroll=1000):
        self.deck = deck or [2,3,4,5,6,7,8,9,10,10,10,10,11]*4*6  # Default: 6 decks
        self.policy = policy  # Function: (player_total, dealer_upcard, is_soft) -> action
        self.bet_size = bet_size
        self.bankroll = bankroll

    def simulate_hand(self, policy=None):
        # Simplified: player and dealer each get two cards, no splits/doubles
        deck = self.deck.copy()
        np.random.shuffle(deck)
        player = [deck.pop(), deck.pop()]
        dealer = [deck.pop(), deck.pop()]
        player_total = sum(player)
        dealer_upcard = dealer[0]
        is_soft = 11 in player and player_total <= 21
        policy = policy or self.policy
        # Player turn
        while player_total < 21:
            action = policy(player_total, dealer_upcard, is_soft) if policy else 'stand'
            if action == 'stand':
                break
            card = deck.pop()
            player.append(card)
            player_total += card
            if 11 in player and player_total > 21:
                player_total -= 10  # Convert ace from 11 to 1
                is_soft = False
        # Dealer turn (hits until 17)
        dealer_total = sum(dealer)
        while dealer_total < 17:
            card = deck.pop()
            dealer.append(card)
            dealer_total += card
            if 11 in dealer and dealer_total > 21:
                dealer_total -= 10
        # Outcome
        if player_total > 21:
            return -self.bet_size
        if dealer_total > 21 or player_total > dealer_total:
            return self.bet_size
        if player_total < dealer_total:
            return -self.bet_size
        return 0

    def run(self, num_hands=100000, policy=None):
        results = []
        bankroll = self.bankroll
        ruin_count = 0
        for _ in range(num_hands):
            profit = self.simulate_hand(policy=policy)
            bankroll += profit
            results.append(profit)
            if bankroll <= 0:
                ruin_count += 1
                bankroll = self.bankroll  # Reset for next simulation
        mean = np.mean(results)
        std = np.std(results)
        edge = mean / self.bet_size
        risk_of_ruin = ruin_count / num_hands
        return {'mean': mean, 'std': std, 'edge': edge, 'risk_of_ruin': risk_of_ruin, 'all_results': results}

    def importance_sampling(self, num_hands=100000, alt_policy=None, base_policy=None):
        # Estimate edge for alt_policy using samples from base_policy
        # For demonstration, use a simple likelihood ratio based on action probabilities
        # (In practice, this requires explicit action probabilities for both policies)
        base_policy = base_policy or self.policy
        alt_policy = alt_policy or self.policy
        weights = []
        rewards = []
        for _ in range(num_hands):
            # For simplicity, assume equal probability for all actions (stub)
            reward = self.simulate_hand(policy=alt_policy)
            rewards.append(reward)
            weights.append(1.0)  # Placeholder: in real IS, use likelihood ratio
        weighted_mean = np.average(rewards, weights=weights)
        return {'weighted_mean': weighted_mean, 'all_rewards': rewards, 'all_weights': weights}

    def stratified_sampling(self, num_hands=100000, strata=10, policy=None):
        # Divide simulations into strata (e.g., by initial player hand value)
        policy = policy or self.policy
        strata_results = [[] for _ in range(strata)]
        for _ in range(num_hands):
            deck = self.deck.copy()
            np.random.shuffle(deck)
            player = [deck.pop(), deck.pop()]
            stratum = min(sum(player) // (22 // strata), strata-1)
            # Simulate hand as usual
            profit = self.simulate_hand(policy=policy)
            strata_results[stratum].append(profit)
        # Aggregate
        means = [np.mean(s) if s else 0 for s in strata_results]
        overall_mean = np.mean([item for sublist in strata_results for item in sublist])
        return {'strata_means': means, 'overall_mean': overall_mean, 'strata_results': strata_results} 