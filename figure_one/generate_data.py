import msprime
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


max_workers = 16


def run(L, N, num_samples, r, model):
    if type(model) == int:
        model = msprime.SmcKApproxCoalescent(hull_offset=model)

    ts = msprime.sim_ancestry(
            samples=num_samples,
            population_size=N,
            sequence_length=L,
            recombination_rate=r,
            model=model,
            ploidy=1)

    return ts

def csv(x):
    return ",".join(map(str, x)) + "\n"

def average_coalescence_time_at_distance_d(params):
    ts, p1, d, n1, n2, rest_params = params
    max_sample_size, L, N, num_samples, r, model, d = rest_params


    left_tree = ts.at(p1)
    right_tree = ts.at(p1+d)

    # Ensure both positions are covered by trees
    if left_tree is None or right_tree is None:
        raise ValueError(
            f"Tree at position {p1} or {p1+d} is None. "
            "Ensure both positions are covered by trees."
        )

    t1 = left_tree.tmrca(n1, n2)
    t2 = right_tree.tmrca(n1, n2)
    if type(model) == int:
        model = f'smck_{model}'
    return [N, num_samples, r, L, model, d, t1, t2]

def process_one_tree_in_parallel(params):
    max_sample_size, L, N, num_samples, r, model, d = params
    results = []
    if num_samples != 2:
        raise ValueError("num_samples must be 2")

    ts = run(L, N, num_samples, r, model)
    positions = np.arange(0, L - d, d)
    innner_tasks = np.array([None] * len(positions), dtype=object)
    for i, p1 in enumerate(positions):
        innner_tasks[i] = (ts, p1, d, 0, 1, params)

    if len(innner_tasks) > max_sample_size:

        innner_tasks = np.random.choice(innner_tasks, max_sample_size, replace=False)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = {executor.submit(average_coalescence_time_at_distance_d, params): params for params in innner_tasks}

        for future in concurrent.futures.as_completed(future_results):
            try:
                results.append(csv(future.result()))
            except Exception as exc:
                params = future_results[future]
                print(f"Task {params} generated an exception: {exc}")

    return results

def process_one_tree_for_one_value(params):
    L, N, num_samples, r, model, d = params

    assert num_samples == 2, "num_samples must be 2"
    assert L==d+1, "L must be equal to d+1"

    ts = run(L, N, num_samples, r, model)
    rest_params = [None, L, N, num_samples, r, model, d]
    result = average_coalescence_time_at_distance_d([ts, 0, L, 0, 1, rest_params])
    return csv(result)

def get_coalescence_times_for_pairs_of_sites_at_distances_d_one_tree_many_values(distances_range=[1, 1e3], models=['hudson', 0, 1, 10, 100, 1000, int(1e4), int(1e5)]):#, 0, 1, 10, 100, 1000, int(1e4), int(1e5)
    num_samples = 2
    sequence_length = 1e6  # total length of genome
    recombination_rate = 1e-8
    population_size=1e6
    outfile = "results.csv"

    repetion = 10
    max_sample_size = int(sequence_length/max(distances_range))

    results = []
    tasks = []
    for d in np.logspace(np.log10(distances_range[0]), np.log10(distances_range[1]), num=4):
        for model in models:
            for _ in range(repetion):
                tasks.append((max_sample_size, sequence_length, population_size, num_samples, recombination_rate, model, d))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = {executor.submit(process_one_tree_in_parallel, params): params for params in tasks}

        for future in concurrent.futures.as_completed(future_results):
            try:
                results.extend(future.result())
            except Exception as exc:
                params = future_results[future]
                print(f"Task {params} generated an exception: {exc}")

    with open(outfile, "w") as f:
        f.write("population_size,num_samples,r,seq_len,model,d,t1,t2\n")
        f.write(''.join(results))
        f.flush()

def get_coalescence_times_for_pairs_of_sites_at_distances_d_one_tree_one_value(distances_range=[1, 1e3],
    models=['hudson', 0, 1, 10, 100, 1000, int(1e4)],
    replicates=10000):

    num_samples = 2
    recombination_rate = 1e-8
    population_size=1e6
    outfile = "results_one_tree.csv"

    ds = [1,10, 50, 100, 500, 1000, 5000, 10000] #+ list(np.logspace(np.log10(distances_range[0]), np.log10(distances_range[1]), num=4))
    results = []
    tasks = []
    for d in ds:
        for model in models:
            for _ in range(replicates):
                d = int(d)
                tasks.append((d+1, population_size, num_samples, recombination_rate, model, d))
    with open(outfile, "w") as f:
        f.write("population_size,num_samples,r,seq_len,model,d,t1,t2\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_results = {executor.submit(process_one_tree_for_one_value, params): params for params in tasks}

            for future in concurrent.futures.as_completed(future_results):
                try:
                    results.extend(future.result())
                    f.write(future.result())
                    f.flush()
                except Exception as exc:
                    params = future_results[future]
                    print(f"Task {params} generated an exception: {exc}")

        f.write(''.join(results))
        f.flush()


if __name__ == "__main__":
    #get_coalescence_times_for_pairs_of_sites_at_distances_d_one_tree_many_values()
    get_coalescence_times_for_pairs_of_sites_at_distances_d_one_tree_one_value()
