'''
 different panels showing the distribution of hull lengths, l1, l2, trapped lengths, and adjusted haplotype lengths
 (they are blue)
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

sample_size = 1000
r = 1e-8
Ne = 1e6
L = 1e6
max_workers=14
max_time = 32497565 * 1.5
filename = f'nodes_distribution_{sample_size}samples_segments.csv'
replicates  = 1
rs = [1e-9]

def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    #plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")

def get_hulls_after_n_generations(params):
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
        #end_time=max_time


    )

    l1 = np.zeros(ts.num_nodes+1)
    l2 = np.zeros(ts.num_nodes+1)
    is_root = np.zeros(ts.num_nodes, dtype=bool)
    for tree in ts.trees():
        l1[(tree.num_children_array == 1)] += tree.span
        l2[(tree.num_children_array == 2)] += tree.span
        root = tree.root
        is_root[root] = True

    assert np.all(ts.samples() == np.arange(ts.num_samples))

    total_span = l1 + l2

    start = np.array([-1] * ts.num_nodes)
    not_started = np.ones(ts.num_nodes, dtype=bool)
    not_started[:ts.sample_size] = False

    for tree in ts.trees():
        tree_nodes = np.zeros(ts.num_nodes, dtype=bool)
        tree_nodes[tree.preorder()] = True

        tree_nodes_did_not_start = np.zeros(ts.num_nodes, dtype=bool)
        tree_nodes_did_not_start[not_started & tree_nodes] = True

        start[tree_nodes_did_not_start] = tree.interval[0]
        not_started[tree_nodes] = False
        if not (not_started[ts.sample_size:].any()):
            break

    end = np.array([-1] * ts.num_nodes)
    not_ended = np.ones(ts.num_nodes, dtype=bool)
    not_ended[:ts.sample_size] = False

    for tree in reversed(ts.trees()):
        tree_nodes = np.zeros(ts.num_nodes, dtype=bool)
        tree_nodes[tree.preorder()] = True

        tree_nodes_did_not_end = np.zeros(ts.num_nodes, dtype=bool)
        tree_nodes_did_not_end[not_ended & tree_nodes] = True

        end[tree_nodes_did_not_end] = tree.interval[1]
        not_ended[tree_nodes] = False
        if not (not_ended[ts.sample_size:].any()):
            break

    youngest_root = np.where(is_root)[0][0]
    hulls = end - start
    df = pd.DataFrame({'node': np.arange(ts.num_nodes), 'hull': hulls, 'l1':l1[:-1], 'l2':l2[:-1], 'total_span': total_span[:-1]})
    keep_rows = np.ones(ts.num_nodes, dtype=bool)
    keep_rows[ts.samples()] = False
    keep_rows[is_root] = False
    keep_rows[np.arange(youngest_root, ts.num_nodes)] = False


    df_spans = pd.DataFrame(0,index=np.arange(ts.num_nodes)[keep_rows], columns=np.arange(ts.num_trees))
    for i, tree in enumerate(ts.trees()):
        tree_nodes = np.zeros(ts.num_nodes, dtype=bool)
        tree_nodes[tree.preorder()] = True
        tree_nodes = tree_nodes[keep_rows]
        df_spans.loc[tree_nodes, i] = tree.interval[1] - tree.interval[0]


    all_segments = []
    no_of_segments = []
    for u in df_spans.index:
        spans = np.array(df_spans.loc[u])
        segments = np.split(spans, np.where(spans == 0)[0])
        #drop segments that only has zeroes
        segments = [seg for seg in segments if not np.all(seg == 0)]
        #drop 0s from each segment
        segments = [seg[seg != 0] for seg in segments]
        all_segments.extend([sum(seg) for seg in segments])
        no_of_segments.append(len(segments))

    avg_all_segments = np.mean(all_segments)
    df = df[keep_rows]
    avg_hulls = df['hull'].mean()
    avg_l1 = df['l1'].mean()
    avg_l2 = df['l2'].mean()
    trapped = df['hull'] - (df['l1'] + df['l2'])
    avg_trapped = trapped.mean()
    avg_no_of_segments = np.mean(no_of_segments)

    oldest_node_time =  ts.nodes()[-1].time
    to_print = lambda x: f'\"{str(list(x))}\"'

    return [Ne, L, _r, _sample_size, str(model), to_print(df['hull']), to_print(df['l1']), to_print(df['l2']), to_print(trapped), to_print(all_segments), to_print(no_of_segments), str(oldest_node_time)]

def generate_data():

    models = ['smc_k', 'Hudson']
    tasks = []

    for _r in rs:
        for model in models:
            for replicate in range(replicates):
                tasks.append((model, sample_size, _r))

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'hulls', 'l1', 'l2', 'trapped', 'adj_hap_len', 'no_of_segments', 'oldest_node_time']))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(get_hulls_after_n_generations, params): params for params in tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result()
                    f.write(csv(result))
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

def plot_distributions_by_model(infile=filename, sample_size=None, L=None):
    # Read CSV
    df = pd.read_csv(infile)

    # Optional filter by sample size
    if sample_size is not None:
        df = df[df['num_samples'] == sample_size]

    # Keep only Hudson and smc_k
    models = ["Hudson", "smc_k"]
    df = df[df["model"].isin(models)]

    # Columns with list-like strings
    list_cols = ['hulls', 'l1', 'l2', 'trapped', 'adj_hap_len']

    # Parse lists
    for col in list_cols:
        df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))

    # Flatten for each model
    data = {model: {} for model in models}
    for model in models:
        sub = df[df["model"] == model]
        for col in list_cols:
            arr = np.concatenate(sub[col].values) if len(sub) > 0 else np.array([])
            if L is not None:
                arr = arr / L
            data[model][col] = arr

    # Plot
    fig, axes = plt.subplots(len(list_cols), 2, figsize=(10, 12), sharex=False)
    fig.suptitle("Distributions of genomic features by model", fontsize=14)

    for row, col in enumerate(list_cols):
        # Determine common x and y limits
        combined = np.concatenate([data[m][col] for m in models if data[m][col].size > 0])
        x_min, x_max = np.min(combined), np.max(combined)

        # Use a common histogram binning for both
        bins = np.linspace(x_min, x_max, 50)

        # Determine common y limit by first computing hist heights
        y_max = 0
        for model in models:
            counts, _ = np.histogram(data[model][col], bins=bins)
            y_max = max(y_max, counts.max())

        for col_idx, model in enumerate(models):
            ax = axes[row, col_idx]
            arr = data[model][col]
            if arr.size > 0:
                ax.hist(arr, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_title(f"{col} - {model}")
            ax.set_ylabel("Count")
            if L is not None:
                ax.set_xlabel("Normalized length")
            else:
                ax.set_xlabel("Length")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max * 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(f'{infile.split(".")[0]}_nodes_distribution')
    plt.close()

if __name__ == "__main__":
    generate_data()
    plot_distributions_by_model(L=L, sample_size=sample_size)
    #plot_stacked_bars()
    #plot_ratio_bars('avg_adj_hap_len')
