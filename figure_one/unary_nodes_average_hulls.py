import msprime
import concurrent
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import time
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore")

sample_size = 2
r = 1e-8
Ne = 1e6
L = 1e6
max_workers=3
filename = f'average_hulls_{sample_size}samples_segments.csv'
models = {'CwR':'Hudson',
          'SMCK(500kb)': msprime.SMCK(500000),
          'SMCK(1)': msprime.SMCK(1),
          'SMCK(0)': msprime.SMCK(0)
          }

rs = [1e-11, 1e-10, 1e-9]


def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    #plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")

def get_hulls_after_n_generations(params):
    model = params[0]; _sample_size = params[1]; _r = params[2]
    model_class = models[model]

    ts = msprime.sim_ancestry(
        samples=_sample_size,
        recombination_rate=_r,
        population_size=Ne,
        sequence_length=L,
        model=model_class,
        #additional_nodes=(msprime.NodeType.COMMON_ANCESTOR),
        coalescing_segments_only=False,
        stop_at_local_mrca=False

    )
    num_trees = ts.num_trees
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
    #keep_rows[np.arange(youngest_root, ts.num_nodes)] = False


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

    return [Ne, L, _r, _sample_size, str(model), str(avg_hulls),
            str(avg_l1), str(avg_l2), str(avg_trapped), str(avg_all_segments),
            str(avg_no_of_segments), str(oldest_node_time), str(num_trees)]

def generate_data(replicates=5):

    tasks = []

    for _r in rs:
        for model in models:
            for replicate in range(replicates):
                tasks.append((model, sample_size, _r))

    timeout_seconds = 3 * 60  # 3 minutes in seconds

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'avg_hulls', 'avg_l1', 'avg_l2', 'avg_trapped', 'avg_adj_hap_len', 'no_of_segments', 'oldest_node_time', 'num_trees']))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(get_hulls_after_n_generations, params): params for params in tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result(timeout=timeout_seconds)
                    f.write(csv(result))
                    f.flush()
                except concurrent.futures.TimeoutError:
                    params = future_results[future]
                    print(f"Task {params} exceeded the timeout of {timeout_seconds} seconds.")                    
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

def plot_single_parameter(param='num_trees', log_f=False, title=None):
    # Read CSV
    import ast

    df = pd.read_csv(filename)
    rs = sorted(df['r'].unique())
    x = np.arange(len(rs))

    ne = df['N'][0]
    assert np.all(df['N'] == ne), "N values are not consistent in the data."

    #models = sorted(df['model'].unique())
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

    width = 0.8 / len(models)  # space bars for each model at the same r

    # Panel 1: avg_l1
    for i, model in enumerate(models):
        df_model = df[df['model'] == model]
        grouped_in = df_model.groupby('r').apply(lambda x: x[param].mean()).reset_index(name=f'mean_{param}')

        color = cmap(i)

        offsets = x + i * width - (width * len(models)) / 2
        ax.bar(offsets, grouped_in[f'mean_{param}'], width,  alpha=0.8, color=color, label=model)
        r_to_x = dict(zip(rs, x))

        scatter_x = df_model['r'].map(r_to_x) + i * width - (width * len(models)) / 2

        ax.scatter(scatter_x,
                df_model[param],
                color=color,
                alpha=0.6,
                s=20,
                linewidth=0.3,
                zorder=3)
    if title is not None:
        plt.title(title, loc='left', fontsize=20)
    else:
        plt.title(f'{param}, sample size {sample_size}, Ne {ne}, L {L}', loc='left', fontsize=20)
    
    human_rho = ((1e-8)*4*1e4)
    plt.grid(True)
    plt.legend(fontsize=16)
    ax.set_xlabel(r'Normalised recombination rate ($\rho / \rho_{\mathrm{human}}$)', fontsize=16)
    x_ticks = [f"{(i*4*ne)/human_rho:.2g}" for i in rs]
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.tick_params(labelsize=16)

    if log_f:
        plt.yscale('log')

        out_file_name = f'single_param{filename.split(".")[0]}_{param}_log'
    else:
        out_file_name = f'single_param{filename.split(".")[0]}_{param}'


    plt.tight_layout()
    save(out_file_name)
    plt.close()

def plot_stacked_bars(infile=filename):
    df = pd.read_csv(infile)

    # Filter on sample size
    df_samples = df[df['num_samples'] == sample_size].copy()


    # Group and aggregate
    grouped = df_samples.groupby(['r', 'model']).agg({
        'avg_hulls': 'mean',
        'avg_l1': 'mean',
        'avg_l2': 'mean',
        'avg_trapped': 'mean',
        'no_of_segments': 'mean'
    }).reset_index()

    '''grouped['trapped'] = grouped.apply(
        lambda row: np.array(row['hulls']) - (np.array(row['l1']) + np.array(row['l2'])),
        axis=1
    )'''
    grouped['avg_hulls'] = grouped['avg_hulls'].apply(lambda x: (x) / L)
    grouped['avg_l1'] = grouped['avg_l1'].apply(lambda x: (x) / L)
    grouped['avg_l2'] = grouped['avg_l2'].apply(lambda x: (x) / L)
    grouped['avg_trapped'] = grouped['avg_trapped'].apply(lambda x: (x) / L)
    # Keep no_of_segments unnormalized for the label
    segments_for_label = grouped['no_of_segments'].copy()
    grouped['no_of_segments'] = grouped['no_of_segments'].apply(lambda x: (x) / L)

    #models = sorted(grouped['model'].unique())
    rs = sorted(grouped['r'].unique())
    x = np.arange(len(rs))
    ne = df['N'][0]
    assert np.all(df['N'] == ne), "N values are not consistent in the data."
    #x = [i* 4 * ne for i in x]

    width = 0.8 / len(models)  # space bars for each model at the same r
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')

    legend_handles = []
    legend_labels = []
    for i, model in enumerate(models):
        color = cmap(i)
        model_data = grouped[grouped['model'] == model]
        model_segments = segments_for_label[grouped['model'] == model]
        offsets = x + i * width - (width * len(models)) / 2
        ax.bar(offsets, model_data['avg_l1'], width,  alpha=0.6)
        ax.bar(offsets, model_data['avg_l2'], width, bottom=model_data['avg_l1'], color=color, alpha=0.9)
        ax.bar(offsets, model_data['avg_trapped'], width,
               bottom=model_data['avg_l1'] + model_data['avg_l2'],
               color='gray', alpha=0.5)
        
        unary_patch = Patch(facecolor=color, alpha=0.6)
        binary_patch = Patch(facecolor=color, alpha=1.0)
        legend_handles.append((binary_patch, unary_patch))
        legend_labels.append(f"{model}  binary / unary")

    legend_handles.append(Patch(facecolor='grey', alpha=1.0))
    legend_labels.append("Trapped material")
    ax.legend(legend_handles, legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=16)

    human_rho = ((1e-8)*4*1e4)
    x_ticks = [f"{(i*4*ne)/human_rho:.2g}" for i in rs]
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(r'Normalised recombination rate ($\rho / \rho_{\mathrm{human}}$)', fontsize=18)
    ax.tick_params(labelsize=16)

    #ax.set_yscale('log')
    ax.set_title('Normalised length (per L)', loc='left', fontsize=18)
    #ax.set_title(f'Stacked bar of l1, l2, and trapped material per r\nSample size {sample_size}, Ne {Ne}, L {L}')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #ax.legend(fontsize=13)
    ax.grid(True)

    plt.tight_layout()
    save(f'{infile.split(".")[0]}_stacked_bar')
    plt.close()

if __name__ == "__main__":
    #generate_data(replicates=10)
    plot_stacked_bars()
    #for param in ['num_trees']:
        #plot_single_parameter(param=param)
    param = 'num_trees'
    title = "Number of trees making the ARG"
    plot_single_parameter(param=param, log_f=True, title=title)
