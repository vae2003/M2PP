import itertools


def _distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def assign_tasks_min_cost(starts, goals):
    """
    Assign goals to starts and return a goal list aligned to starts plus index mapping.
    Uses full permutation search, suitable for small-scale AUV counts.
    """
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have the same length")

    n = len(starts)
    if n == 0:
        return [], []

    best_perm = None
    best_cost = float("inf")
    goal_indices = list(range(n))

    for perm in itertools.permutations(goal_indices):
        total_cost = 0.0
        for i in range(n):
            total_cost += _distance(starts[i], goals[perm[i]])
        if total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm

    assigned_goals = [goals[idx] for idx in best_perm]
    return assigned_goals, list(best_perm)
