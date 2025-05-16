import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = list(plt.cm.tab10.colors) + plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>', '|', '_']

def get_covariance(x, y):
    """
    Calculate the covariance between two lists of numbers.
    """
    #change the data to float
    '''print(x== y)
    print(np.cov(x, y))'''
    return np.cov(x, y)[0,1]
    '''if len(x) != len(y):
        raise ValueError("Lists must have thedata.columns[0] same length.")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    return covariance'''

def get_error_bars_by_bootstrapping(x, y, n=1000):
    """
    Calculate the error bars by bootstrapping.
    """
    if len(x) != len(y):
        raise ValueError("Lists must have the same length.")
    covariance_list = []
    xy = np.array([x, y])
    for i in range(n):
        #sample with replacement
        indices = np.random.choice(len(x), len(x), replace=True)
        sampled_x = xy[0][indices]
        sampled_y = xy[1][indices]
        covariance = get_covariance(sampled_x, sampled_y)
        covariance_list.append(covariance)

    standard_deviation = np.std(covariance_list)

    covariance = np.mean(covariance_list)

    return covariance, 1*standard_deviation

def get_theoretical_value(r, population_size, L):
    R = r*population_size*L #TODO check x2
    return (9+R)/(9+(13*R)+(2*(R**2)))

def plot_bars(file_name='results.csv'):
    """
    Read the CSV file and plot the results.
    """
    data = pd.read_csv(file_name, header=0)
    #change t1 and t2 to float
    data['t1'] = data['t1'].astype(float)/(data['population_size'].astype(float))
    data['t2'] = data['t2'].astype(float)/(data['population_size'].astype(float))


    data = data.sort_values(by=['model', 'd'])
    width = (1/len(data['d'].unique())) # width of each bar
    for i, model in enumerate(sorted(data['model'].unique())):
        for j, d in enumerate(sorted(data['d'].unique())):
            subset = data[(data['model'] == model) & (data['d'] == d)]
            #reindex subset
            t1 = np.array(subset['t1'])
            t2 = np.array(subset['t2'])
            #covariance = get_covariance(t1, t2)
            covariance, covariance_error = get_error_bars_by_bootstrapping(t1, t2)

            theoretical_value = get_theoretical_value(subset['r'].unique()[0], subset['population_size'].unique()[0], subset['d'].unique()[0])
            plt.axhline(y=theoretical_value, color=colors[j], linestyle='--')

            if i==j:
                plt.bar(i+1+(j*width), covariance, width=width, color=colors[j], label=f'd={d}', alpha=0.7)

            else:
                plt.bar(i+1+(j*width), covariance, width=width, color=colors[j], alpha=0.7)

            plt.errorbar(i+1+(j*width), covariance, yerr=covariance_error, fmt='o', color='black', capsize=5, elinewidth=2, alpha=0.7)


    ticks = np.arange(len(data['model'].unique()))+1.5-width/2
    #set x ticks to be the model names
    plt.xticks(ticks, data['model'].unique(), rotation=45)
    plt.xlabel('Model')

    #plt.xlabel('Distance')
    #plt.yscale('log')
    plt.ylabel('Covariance')
    plt.title('Covariance of TMRCA at distance d')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{1}.png")
    plt.savefig(f"figures/{1}.pdf")

def plot_curves(file_name='results_one_tree.csv'):
    data = pd.read_csv(file_name, header=0)
    #change t1 and t2 to float
    data['t1'] = data['t1'].astype(float)/(data['population_size'].astype(float))
    data['t2'] = data['t2'].astype(float)/(data['population_size'].astype(float))

    models = data['model'].unique()
    values = {m:[] for m in models}


    data = data.sort_values(by=['model', 'd'])
    width = (1/len(data['d'].unique())) # width of each bar
    for i, model in enumerate(sorted(models)):
        for j, d in enumerate(sorted(data['d'].unique())):
            if d == 0:
                continue
            subset = data[(data['model'] == model) & (data['d'] == d)]

            t1 = np.array(subset['t1'])
            t2 = np.array(subset['t2'])
            #covariance = get_covariance(t1, t2)
            covariance, covariance_error = get_error_bars_by_bootstrapping(t1, t2)
            theoretical_value = get_theoretical_value(subset['r'].unique()[0], subset['population_size'].unique()[0], subset['d'].unique()[0])
            values[model].append(np.sqrt((theoretical_value-covariance)**2))

    #plot the values as curvers for each model, with error bars
    for i, model in enumerate(sorted(models)):
        plt.plot(data['d'].unique(), values[model], label=model, color=colors[i], marker=markers[i])

    plt.xlabel('Distance')
    plt.xscale('log')
    #plt.yscale('log')
    plt.title('Error in Covariance of TMRCA at distance d')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{2}.png")
    plt.savefig(f"figures/{2}.pdf")



if __name__ == "__main__":
    plot_bars(file_name='results_one_tree.csv')
