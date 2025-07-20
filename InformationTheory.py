import numpy as np
from collections import Counter

def deck_entropy(deck_counts):
    """Compute the entropy (in bits) of the deck given card counts."""
    total = sum(deck_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([count/total for count in deck_counts.values() if count > 0])
    return -np.sum(probs * np.log2(probs))

def update_entropy_sequence(deck_counts, draw_sequence):
    """Given a deck and a sequence of drawn cards, output entropy after each draw."""
    entropies = [deck_entropy(deck_counts)]
    deck = deck_counts.copy()
    for card in draw_sequence:
        if deck[card] > 0:
            deck[card] -= 1
        entropies.append(deck_entropy(deck))
    return entropies

def mutual_information(deck_counts, draw_sequence):
    """Compute the mutual information between the observed sequence and the remaining deck."""
    # For each draw, compute reduction in entropy
    entropies = update_entropy_sequence(deck_counts, draw_sequence)
    total_reduction = entropies[0] - entropies[-1]
    # Mutual information: expected reduction in uncertainty
    return total_reduction

# Example usage:
if __name__ == "__main__":
    # Standard single deck
    deck = Counter({v:4 for v in [2,3,4,5,6,7,8,9,10,'J','Q','K','A']})
    draws = [10, 'A', 2, 5, 10, 'Q', 7]
    print("Initial entropy:", deck_entropy(deck))
    entropies = update_entropy_sequence(deck, draws)
    print("Entropies after each draw:", entropies)
    mi = mutual_information(deck, draws)
    print("Mutual information (total reduction in entropy):", mi) 