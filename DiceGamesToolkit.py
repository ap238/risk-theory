import numpy as np
import itertools
from collections import Counter

class CrapsMarkovChain:
    """
    Markov chain for craps: sequence of dice rolls, pass/don't pass odds, conditional probabilities.
    """
    def __init__(self):
        # States: 'come_out', 'point_4', ..., 'point_10', 'win', 'lose'
        self.states = ['come_out', 'point_4', 'point_5', 'point_6', 'point_8', 'point_9', 'point_10', 'win', 'lose']
        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.P = self._build_transition_matrix()

    def _build_transition_matrix(self):
        probs = Counter(sum(x) for x in itertools.product(range(1,7), repeat=2))
        for k in probs:
            probs[k] /= 36
        P = np.zeros((len(self.states), len(self.states)))
        for s, p in probs.items():
            if s in [7, 11]:
                P[self.state_idx['come_out'], self.state_idx['win']] += p
            elif s in [2, 3, 12]:
                P[self.state_idx['come_out'], self.state_idx['lose']] += p
            elif s in [4,5,6,8,9,10]:
                P[self.state_idx['come_out'], self.state_idx[f'point_{s}']] += p
        for pt in [4,5,6,8,9,10]:
            for s, p in probs.items():
                if s == pt:
                    P[self.state_idx[f'point_{pt}'], self.state_idx['win']] += p
                elif s == 7:
                    P[self.state_idx[f'point_{pt}'], self.state_idx['lose']] += p
                else:
                    P[self.state_idx[f'point_{pt}'], self.state_idx[f'point_{pt}']] += p
        P[self.state_idx['win'], self.state_idx['win']] = 1
        P[self.state_idx['lose'], self.state_idx['lose']] = 1
        return P

    def pass_line_odds(self):
        v = np.zeros(len(self.states))
        v[self.state_idx['come_out']] = 1
        for _ in range(100):
            v = v @ self.P
        return v[self.state_idx['win']], v[self.state_idx['lose']]

    def simulate(self, n=10000):
        wins = 0
        results = []
        for _ in range(n):
            state = 'come_out'
            rolls = 0
            while state not in ['win', 'lose']:
                idx = self.state_idx[state]
                state = np.random.choice(self.states, p=self.P[idx])
                rolls += 1
            if state == 'win':
                wins += 1
                results.append(1)
            else:
                results.append(0)
        win_prob = wins/n
        lose_prob = 1-win_prob
        mean = np.mean(results)
        var = np.var(results)
        return {'win_prob': win_prob, 'lose_prob': lose_prob, 'mean': mean, 'variance': var, 'results': results}

class SicBoOdds:
    """
    Sic Bo: Combinatorial enumeration for all bet types, odds calculator, and Monte Carlo simulation.
    """
    @staticmethod
    def all_outcomes():
        return list(itertools.product(range(1,7), repeat=3))

    @staticmethod
    def bet_probability(bet_type, value=None):
        outcomes = SicBoOdds.all_outcomes()
        total = len(outcomes)
        if bet_type == 'triple':
            count = sum(1 for o in outcomes if o[0]==o[1]==o[2]==value)
        elif bet_type == 'sum':
            count = sum(1 for o in outcomes if sum(o)==value)
        elif bet_type == 'any_triple':
            count = sum(1 for o in outcomes if o[0]==o[1]==o[2])
        elif bet_type == 'double':
            count = sum(1 for o in outcomes if len(set(o))==2 and (o.count(value)==2))
        elif bet_type == 'single':
            count = sum(1 for o in outcomes if value in o)
        else:
            count = 0
        return count/total

    @staticmethod
    def odds_display():
        display = {}
        for s in range(4,18):
            display[f'sum_{s}'] = SicBoOdds.bet_probability('sum', s)
        for v in range(1,7):
            display[f'triple_{v}'] = SicBoOdds.bet_probability('triple', v)
            display[f'double_{v}'] = SicBoOdds.bet_probability('double', v)
            display[f'single_{v}'] = SicBoOdds.bet_probability('single', v)
        display['any_triple'] = SicBoOdds.bet_probability('any_triple')
        return display

    @staticmethod
    def monte_carlo_sim(bet_type, value=None, n=100000):
        # Simulate n rounds, return empirical probability, mean, variance
        wins = 0
        results = []
        for _ in range(n):
            roll = tuple(np.random.randint(1,7) for _ in range(3))
            if bet_type == 'triple' and roll[0]==roll[1]==roll[2]==value:
                wins += 1
                results.append(1)
            elif bet_type == 'sum' and sum(roll)==value:
                wins += 1
                results.append(1)
            elif bet_type == 'any_triple' and roll[0]==roll[1]==roll[2]:
                wins += 1
                results.append(1)
            elif bet_type == 'double' and len(set(roll))==2 and roll.count(value)==2:
                wins += 1
                results.append(1)
            elif bet_type == 'single' and value in roll:
                wins += 1
                results.append(1)
            else:
                results.append(0)
        prob = wins/n
        mean = np.mean(results)
        var = np.var(results)
        return {'prob': prob, 'mean': mean, 'variance': var, 'results': results}

class YahtzeeDP:
    """
    Yahtzee: Scoring combinations, expected value, dynamic programming for optimal play, and Monte Carlo simulation.
    """
    def __init__(self):
        self.memo = {}

    def expected_value(self, dice, rolls_left, used_categories):
        key = (tuple(sorted(dice)), rolls_left, tuple(sorted(used_categories)))
        if key in self.memo:
            return self.memo[key]
        if rolls_left == 0:
            return self.best_score(dice, used_categories)
        best = 0
        for keep in itertools.product([True, False], repeat=5):
            kept = tuple(d for d, k in zip(dice, keep) if k)
            reroll = 5 - len(kept)
            if reroll == 0:
                val = self.expected_value(kept, 0, used_categories)
            else:
                exp = 0
                for new in itertools.product(range(1,7), repeat=reroll):
                    new_dice = kept + new
                    exp += self.expected_value(new_dice, rolls_left-1, used_categories)
                val = exp / (6**reroll)
            best = max(best, val)
        self.memo[key] = best
        return best

    def best_score(self, dice, used_categories):
        available = [c for c in range(1,7) if c not in used_categories]
        if not available:
            return 0
        return max(dice.count(c)*c for c in available)

    def monte_carlo_sim(self, n=10000):
        # Simulate n Yahtzee hands, return empirical mean and variance of best upper-section score
        results = []
        for _ in range(n):
            dice = tuple(np.random.randint(1,7) for _ in range(5))
            val = self.expected_value(dice, 2, set())
            results.append(val)
        mean = np.mean(results)
        var = np.var(results)
        return {'mean': mean, 'variance': var, 'results': results}

class CeeLoOdds:
    """
    Cee-lo: Probability distributions for triplets, pairs, and sophisticated odds display, with Monte Carlo simulation.
    """
    @staticmethod
    def outcome_distribution():
        outcomes = list(itertools.product(range(1,7), repeat=3))
        dist = {'triplet':0, 'pair':0, '4-5-6':0, '1-2-3':0, 'other':0}
        for o in outcomes:
            if o[0]==o[1]==o[2]:
                dist['triplet'] += 1
            elif sorted(o)==[1,2,3]:
                dist['1-2-3'] += 1
            elif sorted(o)==[4,5,6]:
                dist['4-5-6'] += 1
            elif len(set(o))==2:
                dist['pair'] += 1
            else:
                dist['other'] += 1
        total = len(outcomes)
        return {k: v/total for k,v in dist.items()}

    @staticmethod
    def odds_display():
        return CeeLoOdds.outcome_distribution()

    @staticmethod
    def monte_carlo_sim(n=100000):
        # Simulate n Cee-lo rolls, return empirical distribution
        dist = {'triplet':0, 'pair':0, '4-5-6':0, '1-2-3':0, 'other':0}
        for _ in range(n):
            o = tuple(np.random.randint(1,7) for _ in range(3))
            if o[0]==o[1]==o[2]:
                dist['triplet'] += 1
            elif sorted(o)==[1,2,3]:
                dist['1-2-3'] += 1
            elif sorted(o)==[4,5,6]:
                dist['4-5-6'] += 1
            elif len(set(o))==2:
                dist['pair'] += 1
            else:
                dist['other'] += 1
        total = sum(dist.values())
        for k in dist:
            dist[k] /= total
        return dist

# Example usage:
if __name__ == "__main__":
    print("--- Craps ---")
    craps = CrapsMarkovChain()
    win, lose = craps.pass_line_odds()
    print(f"Pass line win: {win:.4f}, lose: {lose:.4f}")
    sim = craps.simulate(10000)
    print(f"Simulated win: {sim['win_prob']:.4f}, lose: {sim['lose_prob']:.4f}, mean: {sim['mean']:.4f}, variance: {sim['variance']:.4f}")

    print("\n--- Sic Bo ---")
    print("Sic Bo odds:")
    for k, v in SicBoOdds.odds_display().items():
        print(f"{k}: {v:.4f}")
    sim = SicBoOdds.monte_carlo_sim('sum', 10, n=10000)
    print(f"Monte Carlo for sum 10: prob={sim['prob']:.4f}, mean={sim['mean']:.4f}, variance={sim['variance']:.4f}")

    print("\n--- Yahtzee ---")
    yahtzee = YahtzeeDP()
    dice = (1,2,3,4,5)
    exp_val = yahtzee.expected_value(dice, 2, set())
    print(f"Expected value for dice {dice} with 2 rolls left: {exp_val:.2f}")
    sim = yahtzee.monte_carlo_sim(n=1000)
    print(f"Monte Carlo Yahtzee: mean={sim['mean']:.2f}, variance={sim['variance']:.2f}")

    print("\n--- Cee-lo ---")
    print("Cee-lo odds:")
    for k, v in CeeLoOdds.odds_display().items():
        print(f"{k}: {v:.4f}")
    sim = CeeLoOdds.monte_carlo_sim(n=10000)
    print("Monte Carlo Cee-lo distribution:")
    for k, v in sim.items():
        print(f"{k}: {v:.4f}") 