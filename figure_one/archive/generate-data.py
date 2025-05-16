import msprime
import numpy as np

# Parameters
sample_size = 10
sequence_length = 1e5  # total length of genome
recombination_rate = 1e-3
mutation_rate = 0  # not needed here
distance = 1000  # distance between sites (in bp)
num_replicates = 1000  # to average over simulations

def average_coalescence_time_at_distance_d(ts, d):
    times = []
    positions = np.arange(0, ts.sequence_length - d, d)
    for pos in positions:
        left_ts = ts.at(pos)
        right_ts = ts.at(pos + d)

        for tree in left_ts.trees():
            if tree in right_ts:
                print(f"Tree {tree} is in both left and right trees")

        # Ensure both positions are covered by trees
        if left_tree is None or right_tree is None:
            continue

        # Pick pairs of samples
        for i in range(0, ts.num_samples, 2):
            if i + 1 >= ts.num_samples:
                break
            n1, n2 = i, i + 1

            # Get TMRCA at the two positions
            t1 = left_tree.tmrca(n1, n2)
            t2 = right_tree.tmrca(n1, n2)

            # Average TMRCA for the pair of positions
            times.append((t1 + t2) / 2)

    return np.mean(times) if times else np.nan

def _average_coalescence_time_at_distance_d(ts, d):
    times = []
    positions = np.arange(0, ts.sequence_length - d, d)
    for pos in positions:
        left_tree = ts.at(pos)
        right_tree = ts.at(pos + d)

        # Ensure both positions are covered by trees
        if left_tree is None or right_tree is None:
            continue

        # Pick pairs of samples
        for i in range(0, ts.num_samples, 2):
            if i + 1 >= ts.num_samples:
                break
            n1, n2 = i, i + 1

            # Get TMRCA at the two positions
            t1 = left_tree.tmrca(n1, n2)
            t2 = right_tree.tmrca(n1, n2)

            # Average TMRCA for the pair of positions
            times.append((t1 + t2) / 2)

    return np.mean(times) if times else np.nan

# Run simulation
avg_tmrcas = []
for _ in range(num_replicates):
    ts = msprime.sim_ancestry(
        samples=sample_size,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=None
    )
    avg_tmrcas.append(average_coalescence_time_at_distance_d(ts, distance))

# Filter out None values
avg_tmrcas = [tmrca for tmrca in avg_tmrcas if tmrca is not None]
overall_average = np.nanmean(avg_tmrcas)
print(f"Average coalescence time for site pairs {distance} bp apart: {overall_average}")
