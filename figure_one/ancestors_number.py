import msprime
import concurrent
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

sample_size = 10000
r = 1e-9
Ne = 1e6
L = 1e6
filename = 'lineages.csv'

models = {'Hudson':'Hudson',
          'SMCK(1)': msprime.SMCK(1),
          }

def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.pdf")

def get_ancestors_after_n_generations(params):
    model = params[0]; generations = params[1]
    
    model_class  = models[model]
    sim = msprime.ancestry._parse_simulate(
        sample_size=sample_size,
        recombination_rate=r,
        Ne=Ne,
        length=L,
        model=model_class,
        end_time=generations,
    )
    sim.run()
    ancestors = len(sim.ancestors)
    finish_time = sim.time
    return [Ne, L, r, sample_size, str(model), generations, finish_time, ancestors]

def generate_data(gens_list=[1,10,100,1000,10000,100000,100000,1000000, None], reps=25):
    tasks = []
    for gen in gens_list:
        for model in models.keys():
            for rep in range(reps):
                tasks.append((model, gen))

    with open(filename, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'gens', 'finish_time', 'ancestors']))


        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            # Submit all tasks and get futures
            future_results = {executor.submit(get_ancestors_after_n_generations, params): params for params in tasks}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    result = future.result()
                    f.write(csv(result))
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

def plot():
    df = pd.read_csv(filename)

    models = sorted(df['model'].unique())



    # Define a colormap for sample sizes
    cmap = plt.get_cmap("viridis").resampled(len(models))

    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        subset = df[df['model'] == model]
        subset = subset[subset['ancestors'] != 0]
        grouped_data = subset.groupby('gens')['ancestors'].mean().reset_index()
        grouped_data = grouped_data.sort_values('gens')
        xs = list(grouped_data['gens'])
        ys = list(grouped_data['ancestors'])

        # deal with simulations to the end of time
        subset = df[df['gens'].isna()]
        grouped_data = (
                    subset[subset['model'] == model]
                    .groupby('model')[['ancestors', 'finish_time']]
                    .mean()
                    .reset_index()
                )
        grouped_data = grouped_data.sort_values('finish_time')

        xs += list(grouped_data['finish_time'])
        ys += list(grouped_data['ancestors'])


        plt.plot(xs, ys, linestyle='-', color=cmap(i),
                 linewidth=2, marker='o', markersize=5, label=f'{model}', alpha=0.5)


    plt.xscale('log')
    #plt.yscale('log')

    # Add labels and title
    plt.xlabel('generations')
    plt.title(f'Number of Ancestors after n generations. Ne:{Ne}, L:{L}, r:{r}, num_samples:{sample_size}')

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    save(filename.split('.')[0])
    plt.close()
    #plt.show()

if __name__ == "__main__":
    gens_list= np.logspace(0, 4, num=8, dtype=int).tolist() + \
        np.logspace(4, 6, num=6, dtype=int).tolist() +\
        np.logspace(6, 7, num=5, dtype=int).tolist()  + [None]
    generate_data(gens_list=gens_list, reps=25)
    plot()
