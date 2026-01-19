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

max_workers=14
filename = f'speed.csv'
replicates  = 25
Ne = 1e6

seq_len_dor = 23513712
recombination_rate = 1e-8 #2.40463e-08
sample_sizes = [2, 10, 100, 1000, 10000]

lengths = np.logspace(1, 7, num=7, dtype=int)
lengths = np.append(lengths, seq_len_dor)
shortened_lengths = lengths[lengths <= 1e5]
models = {'Hudson':'Hudson',
          'smc(k=500k)': msprime.SmcKApproxCoalescent(hull_offset=500000),
          'smc(k=1)': msprime.SmcKApproxCoalescent(hull_offset=1),
          'smc(k=0)': msprime.SmcKApproxCoalescent(hull_offset=0)
          }

models_for_sample_size = {'Hudson':'Hudson',
                          'smc(k=1)': msprime.SmcKApproxCoalescent(hull_offset=1)}

neL = Ne * lengths
shortened_neL = Ne * shortened_lengths
drosophila_neL = Ne * seq_len_dor
human_neL = 10**4 * 248956422

def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    #plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")

def get_exc_time(params):
    model = params[0]; length = params[1]; sample_size = params[2]
    model_class = models[model]
    start_time = time.time()
    ts = msprime.sim_ancestry(
        samples=sample_size,
        ploidy=1,
        sequence_length=length,
        recombination_rate=recombination_rate,
        population_size=Ne,
        model=model_class
    )
    ex_time = time.time() - start_time

    return [Ne, length, recombination_rate, sample_size, str(model), ex_time]

def generate_data():

    tasks = []

    #loop for speed test per model and per length
    for model in models:
        for replicate in range(replicates):
            if model not in ['smc(k=1)', 'smc(k=0)']:
                allowed_L = shortened_lengths
            else:
                allowed_L = lengths

            for length in allowed_L:

                if model in models_for_sample_size:
                    for sample_size in sample_sizes:
                        tasks.append((model, length, sample_size))
                else:
                    tasks.append((model, length, 2))

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'ex_time']))

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

def plot_speed_per_model(infile=filename):
    df = pd.read_csv(infile)
    df = df[df['num_samples']==2]

    df_avg = df.groupby(['model', 'L']).mean().reset_index()

    hudson_times = df_avg[df_avg['model']=='Hudson']['ex_time'].to_numpy()
    fit_times = np.polyfit(shortened_neL, hudson_times, 2)
    fit_fn = np.poly1d(fit_times)
    np.log10(fit_fn(neL[-1]))
    fitted_line = fit_fn(neL[1:])

    for model in models:
        model_times = df_avg[df_avg['model']==model]['ex_time'].to_numpy()
        xs = neL[:len(model_times)]
        plt.plot(xs, model_times, marker='o', label=model)

    plt.plot(neL[1:], fitted_line, linestyle='--', color='gray', label='Quadratic fit (Hudson)')

    plt.axvline(x=drosophila_neL, color='green', linestyle=':', label='Drosophila Ne*L (chrom 2L)', linewidth=4)
    plt.axvline(x=human_neL, color='purple', linestyle=':', label='Human Ne*L (chrom 1)', linewidth=4)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Population-scaled Sequence Length (Ne * L)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('SMC(k) vs Hudson Execution Time')
    plt.legend()
    #plt.grid(True, which="both", ls="--")
    save('speed_per_model')
    plt.clf()

def plot_speed_per_sample_size(infile=filename):
    df = pd.read_csv(infile)
    df = df[df['model'].isin(models_for_sample_size.keys())]

    df_avg = df.groupby(['model', 'L', 'num_samples']).mean().reset_index()

    for model in models_for_sample_size:
        if model not in ['smc(k=1)']: continue
        for sample_size in sample_sizes:
            model_times = df_avg[(df_avg['model']==model) & (df_avg['num_samples']==sample_size)]['ex_time'].to_numpy()
            xs = neL[:len(model_times)]
            plt.plot(xs, model_times, marker='o', label=f"{model}, n={sample_size}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Population-scaled Sequence Length (Ne * L)')
    #plt.ylabel('Execution Time (seconds)')
    plt.title('SMC(k=1) execution time (seconds) for different sample sizes.')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    save('speed_per_sample_size')
    plt.clf()


if __name__ == "__main__":
    generate_data()
    plot_speed_per_model()
    plot_speed_per_sample_size()
