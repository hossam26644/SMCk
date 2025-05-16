import msprime

sample_size = 10000
r = 1e-8
Ne = 1e6
L = 1e6
model = msprime.SmcKApproxCoalescent()
model = 'Hudson'
#model = 'dtwf'

def get_ancestors_after_n_generations(model=model, generations=10):
    sim = msprime.ancestry._parse_simulate(
        sample_size=sample_size,
        recombination_rate=r,
        Ne=Ne,
        length=L,
        model=model,
        end_time=generations,
    )
    sim.run()
    ancestors = len(sim.ancestors)
    print(f"Number of ancestors after {generations} generations: {ancestors}")

if __name__ == "__main__":
    get_ancestors_after_n_generations()