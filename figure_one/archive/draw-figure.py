import msprime
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

# Parameters
sample_size = 10
sequence_length = 1e6  # total length of genome
recombination_rate = 1e-2
num_replicates = 100

def simulate_tmrcas(model):
    ts = msprime.sim_ancestry(
        samples=sample_size,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        model=model,
    )
    # Return the TMRCA for all trees in this replicate
    return [tree.time(tree.root) for tree in ts.trees()]

def parallel_tmrcas(model):
    tmrcas = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(simulate_tmrcas, [model] * num_replicates)
        for tmrcas_per_replicate in results:
            tmrcas.extend(tmrcas_per_replicate)
    return tmrcas

# Standard model
tmrcas_standard = parallel_tmrcas(msprime.StandardCoalescent())  # standard coalescent
print(f"Standard model finished")
# SMC' model
tmrcas_smck = parallel_tmrcas(msprime.SmcKApproxCoalescent())

# Box plot
plt.boxplot([tmrcas_standard, tmrcas_smck], labels=["Standard", "SMC'"], showfliers=False)
plt.ylabel("TMRCA")
plt.title("Distribution of Coalescence Times")
plt.show()