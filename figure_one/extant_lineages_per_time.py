'''
blue two panels never used
'''

import msprime
import concurrent
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore")

sample_size = 2
r = 1e-10
Ne = 1e6
L = 1e6
max_workers=14
max_time = 32497565 * 1.5
filename = f'lineages_per_generation_{sample_size}samples_segments.csv'
replicates  = 1
rs = [1e-10]
end_time = 100000

def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    #plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")

def get_lineages_per_generation(params):
    model = params[0]; _sample_size = params[1]; _r = params[2]
    if model=='smc_k': inner_model=msprime.SmcKApproxCoalescent()
    else: inner_model=model

    if type(model) == int: inner_model= msprime.SmcKApproxCoalescent(hull_offset=model)
    ts = msprime.sim_ancestry(
        samples=_sample_size,
        recombination_rate=_r,
        population_size=Ne,
        sequence_length=L,
        model=inner_model,
        #additional_nodes=(msprime.NodeType.COMMON_ANCESTOR),
        coalescing_segments_only=False,
        #stop_at_local_mrca=False,
        end_time=end_time


    )
    lineages = np.zeros(end_time + 1, dtype=int)
    for tree in ts.trees():
        for i in range(lineages.size):
            lineages[i] += tree.num_lineages(i)
    to_print = lambda x: f'\"{str(list(x))}\"'

    return [Ne, L, _r, _sample_size, str(model), to_print(lineages)]

def generate_data():

    models = ['smc_k', 'Hudson']
    tasks = []

    for _r in rs:
        for model in models:
            for replicate in range(replicates):
                tasks.append((model, sample_size, _r))

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'lineages']))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(get_lineages_per_generation, params): params for params in tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result()
                    f.write(csv(result))
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

def plot_lineages_by_model(infile=filename):
    # Read CSV
    df = pd.read_csv(infile)

    # Models we care about
    models = ["Hudson", "smc_k"]

    # Parse list-like strings into numeric arrays
    df["lineages"] = df["lineages"].apply(lambda x: np.array(ast.literal_eval(x), dtype=int))

    # Create figure: 1 row × 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle("Lineages along the genome", fontsize=14)

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue

        # Assuming the positions are equally spaced along the sequence
        # Create an x-axis: divide the genome into len(lineages) segments
        L = sub.iloc[0]["L"]
        lineage_array = sub.iloc[0]["lineages"]
        positions = np.linspace(1, len(lineage_array), len(lineage_array))

        ax.plot(positions, lineage_array, marker="o", markersize=3, lw=1)
        ax.set_title(model)
        ax.set_xlabel("generation")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Number of lineages")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
if __name__ == "__main__":
    generate_data()
    plot_lineages_by_model()
    #plot_ratio_bars('avg_adj_hap_len')
