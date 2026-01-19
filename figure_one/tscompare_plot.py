import msprime
import concurrent
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import ast
import tscompare
import warnings
warnings.filterwarnings("ignore")

max_workers=14
filename = f'diff_tscompare.csv'
replicates  = 20
Ne = 1e6
sample_size = 4
recombination_rate = 1e-9 #2.40463e-08
seq_len = 1e6
models = {'Hudson':'Hudson',
          'k=500k': msprime.SmcKApproxCoalescent(hull_offset=500000),
          'k=100k': msprime.SmcKApproxCoalescent(hull_offset=100000),
          'k=1': msprime.SmcKApproxCoalescent(hull_offset=1),
          'k=0': msprime.SmcKApproxCoalescent(hull_offset=0),
          'k=10': msprime.SmcKApproxCoalescent(hull_offset=10),           
          'k=100': msprime.SmcKApproxCoalescent(hull_offset=100),           
          'k=1k': msprime.SmcKApproxCoalescent(hull_offset=1000),           
          'k=10k': msprime.SmcKApproxCoalescent(hull_offset=10000),           
          }
models_ordered = ['Hudson', 'k=0', 'k=1','k=10', 'k=100','k=1k', 'k=10k', 'k=100k', 'k=500k']



def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    #plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")

def get_exc_time(params):
    model = params[0]
    model_class = models[model]
    start_time = time.time()
    ts_hudson = msprime.sim_ancestry(
        samples=sample_size,
        ploidy=1,
        sequence_length=seq_len,
        recombination_rate=recombination_rate,
        population_size=Ne,
        coalescing_segments_only=False,
    )
    ts_model = msprime.sim_ancestry(
        samples=sample_size,
        ploidy=1,
        sequence_length=seq_len,
        recombination_rate=recombination_rate,
        population_size=Ne,
        model=model_class,
        coalescing_segments_only=False,
    )
    node_times_matched, _span, best_id  = tscompare.match_node_ages(ts_hudson, ts_model)
    all_smc_node_times = np.array([n.time for n in ts_model.nodes()])
    node_times_smc = all_smc_node_times[best_id]

    mask = ~np.isnan(node_times_matched) & ~np.isnan(node_times_smc)
    node_times_matched = node_times_matched[mask]
    node_times_smc = node_times_smc[mask]
    assert (node_times_matched == node_times_smc).all()

    node_times_hudson = np.array([n.time for n in ts_hudson.nodes()])
    node_times_hudson = node_times_hudson[mask]
    assert len(node_times_hudson) == len(node_times_matched)
    diff = np.log(1 + node_times_hudson[sample_size:]) - np.log(1+ node_times_matched[sample_size:])
    masked_span = _span[mask][sample_size:]
    diff_by_span = (diff * masked_span)/seq_len
    rmse = np.sqrt(np.mean(diff_by_span**2))
    

    x= np.sqrt(
            np.sum(diff ** 2 * masked_span)
            / np.sum(masked_span)
        )

    dis = tscompare.haplotype_arf(ts_hudson, ts_model)
    return [Ne, seq_len, recombination_rate, sample_size, str(model), dis.arf, dis.tpr, dis.matched_span[0], dis.matched_span[1], dis.rmse, f'\"{diff.tolist()}\"', rmse]

def generate_data():

    tasks = []

    #loop for speed test per model and per length
    for model in models:
        for replicate in range(replicates):
            tasks.append((model,))

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'arf', 'tpr', 'matched_span', 'inverse_matched_span', 'rmse', 'time_diffs', 'calc_rmse']))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(get_exc_time, params): params for params in tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result()
                    f.write(csv(result))
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")
                    raise exc

def plot(infile=filename):
    df = pd.read_csv(infile)

    df['model'] = pd.Categorical(df['model'], categories=models_ordered, ordered=True)
    df['time_diffs'] = df['time_diffs'].apply(ast.literal_eval)

    #a box plot of arf per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    #df.boxplot(column='arf', by='model', ax=ax)
    sns.violinplot(data=df, x='model', y='arf', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'arf'].median()
    ax.axhline(mean_hudson, color='red', linestyle='--')    
    plt.title('TSCompare Robinson-Foulds relative dissimilarity')
    plt.suptitle('')
    plt.ylabel('Average RF Distance (ARF)')
    save('tscompare_accuracy_comparison')

    #a box plot of tpr per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='tpr', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'tpr'].median()
    ax.axhline(mean_hudson, color='red', linestyle='--')        
    plt.title('TSCompare true proportion represented Comparison')
    plt.suptitle('')
    plt.ylabel('true proportion represented (TPR)')
    save('tscompare_tpr_comparison')

    #a box plot of rmse per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='rmse', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'rmse'].median()
    ax.axhline(mean_hudson, color='red', linestyle='--')
    plt.title('TSCompare RMSE Comparison')
    plt.suptitle('')
    plt.ylabel('Root Mean Square Error (RMSE)')
    save('tscompare_rmse_comparison')

    df['sum_matched_span'] = df['matched_span'] + df['inverse_matched_span']
    #a box plot of matched span length per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='sum_matched_span', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'sum_matched_span'].median()
    ax.axhline(mean_hudson, color='red', linestyle='--')     
    plt.title('TSCompare Matched Span Length Comparison')
    plt.suptitle('')
    plt.ylabel('Matched Span Length + inverse_match')
    save('tscompare_sum_matched_span_length_comparison')

    #a box plot of inverse matched span length per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='inverse_matched_span', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'inverse_matched_span'].median()   
    ax.axhline(mean_hudson, color='red', linestyle='--')     
    plt.title('TSCompare Matched Span Length Comparison')
    plt.suptitle('')
    plt.ylabel('Inverse_match matched Span Length')
    save('tscompare_inverse_matched_span_length_comparison')

    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='matched_span', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'matched_span'].median()   
    ax.axhline(mean_hudson, color='red', linestyle='--')       
    plt.title('TSCompare Matched Span Length Comparison')
    plt.suptitle('')
    plt.ylabel('Matched Span Length')
    save('tscompare_matched_span_length_comparison')

    grouped = df.groupby('model').agg({'time_diffs': lambda x: sum(x, [])}).reset_index()
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    data_to_plot = grouped['time_diffs'].tolist()  
    labels = grouped['model'].tolist()      
    sns.violinplot(data_to_plot, ax=ax, alpha=0.95, palette='Set3')
    plt.xticks(ticks=range(len(labels)), labels=labels)

    plt.title('TSCompare time differences per node')
    plt.suptitle('')
    plt.ylabel('Time Differences')
    save('tscompare_time_differences_comparison')

    #a box plot of rmse per model
    plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1)
    sns.violinplot(data=df, x='model', y='calc_rmse', ax=ax, alpha=0.95, palette='Set3') 
    mean_hudson = df.loc[df['model'] == 'Hudson', 'calc_rmse'].median()
    ax.axhline(mean_hudson, color='red', linestyle='--')
    plt.title('TSCompare RMSE Comparison')
    plt.suptitle('')
    plt.ylabel('Root Mean Square Error (RMSE)')
    save('tscompare_calc_rmse_comparison')        

if __name__ == "__main__":
    
    generate_data()
    plot()
