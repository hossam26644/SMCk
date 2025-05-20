import msprime
import concurrent
import pandas as pd
import matplotlib.pyplot as plt


sample_size = 10000
r = 1e-8
Ne = 1e6
L = 1e6
#model = msprime.SmcKApproxCoalescent()
#model = 'Hudson'
#model = 'dtwf'

def csv(x):
    return ",".join(map(str, x)) + "\n"

def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


def get_ancestors_after_n_generations(params):
    model = params[0]; generations = params[1]
    if model=='smc_k': inner_model=msprime.SmcKApproxCoalescent()
    else: inner_model=model

    if type(model) == int: inner_model= msprime.SmcKApproxCoalescent(hull_offset=model)
    sim = msprime.ancestry._parse_simulate(
        sample_size=sample_size,
        recombination_rate=r,
        Ne=Ne,
        length=L,
        model=inner_model,
        end_time=generations,
    )
    sim.run()
    ancestors = len(sim.ancestors)
    return [Ne, L, r, sample_size, str(model), generations, ancestors]

def generate_data(gens_list=[1,5,10,50,100,500,1000], reps=25, outfile='lineages.csv'):
    models = [10, 100, 1000, 10000, 100000, 1000000, 'smc_k', 'Hudson', 'dtwf']
    tasks = []
    for gen in gens_list:
        for model in models:
            for rep in range(reps):
                tasks.append((model, gen))

    with open(outfile, "w") as f:
        f.write(csv(['N', 'L', 'r','num_samples', 'model', 'gens', 'ancestors']))


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

def plot(infile='lineages.csv'):
    df = pd.read_csv(infile)

    models = sorted(df['model'].unique())

    # Define a colormap for sample sizes
    cmap = plt.cm.get_cmap("viridis", len(models))

    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):
        subset = df[df['model'] == model]
        grouped_data = subset.groupby('gens')['ancestors'].mean().reset_index()
        grouped_data = grouped_data.sort_values('gens')
        plt.plot(grouped_data['gens'], grouped_data['ancestors'], linestyle='-', color=cmap(i),
                 linewidth=2, marker='o', markersize=5, label=f'{model}', alpha=0.5)

    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('generations')
    plt.title(f'Number of Ancestors after n generations. Ne:{Ne}, L:{L}, r:{r}, num_samples:{sample_size}')

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Save the plot
    plt.tight_layout()
    save(infile.split('.')[0])
    plt.close()
    #plt.show()

if __name__ == "__main__":
    plot()
    #generate_data()
    #get_ancestors_after_n_generations()