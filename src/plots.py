import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def univariate_plot(var1, ax, df, positive_class='human'):
    g = df.groupby([var1, 'cause_group'], as_index=False)
    counts = g.count()
    counts = counts[[var1, 'cause_group', 'state']]
    counts = counts.pivot_table(values='state', index=var1, columns='cause_group')
    counts['prob_human'] = counts[positive_class]/(counts['other']+counts['human'])

    ax.plot(counts.index, counts['prob_human'], label='Probability of {} cause'.format(positive_class), color='b', linewidth=4)
    ax.bar(counts.index, (counts['human']+counts['other'])/(counts['human'].sum()+counts['other'].sum()),
            label='Fraction of fires')
    ax.set_xlabel(var1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize='x-large')
    ax.set_title("Probability of human cause vs {}".format(var1))



def univariate_binned_plot(bins, var1, ax, df, positive_class='human'):
    binned_col = "{}_binned".format(var1)
    df[binned_col] = pd.cut(df[var1], bins)


    g = df.groupby([binned_col, 'cause_group'], as_index=False)
    counts = g.count()
    counts = counts[[binned_col, 'cause_group', 'state']]
    counts = counts.pivot_table(values='state', index=binned_col, columns='cause_group')
    counts['prob_human'] = counts[positive_class]/(counts['other']+counts['human'])

    ax.plot(np.arange(len(counts.index)), counts['prob_human'], label='Probability of {} cause'.format(positive_class), c='b', linewidth=4)
    ax.bar(np.arange(len(counts.index)), (counts['human']+counts['other'])/(counts['human'].sum()+counts['other'].sum()), label='Fraction of fires')
    ax.legend(loc=0, fontsize='xx-large')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(var1, fontsize=24)
    ax.set_xticklabels(bins[:-1])
    ax.set_xticks(np.arange(0,len(bins)-1,1))

def univariate_binned_counts_plot(bins, var1, ax, df, positive_class='human'):
    binned_col = "{}_binned".format(var1)
    df[binned_col] = pd.cut(df[var1], bins)

    n_years = len(df['fire_year'].unique())
    g = df.groupby([binned_col, 'cause_group'], as_index=False)
    counts = g.count()
    counts = counts[[binned_col, 'cause_group', 'state']]
    counts = counts.pivot_table(values='state', index=binned_col, columns='cause_group')
    counts['prob_human'] = counts[positive_class]/(counts['other']+counts['human'])

    ax.plot(np.arange(len(counts.index)), counts['other']/n_years, label='Natural cause', c='b', linewidth=4)
    ax.plot(np.arange(len(counts.index)), counts['human']/n_years, label='Human cause', c='r', linewidth=4)
    ax.legend(loc=0, fontsize='xx-large')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(var1, fontsize=24)
    ax.set_ylabel('Number of fires per year', fontsize=24)
    ax.set_xticklabels(bins[:-1])
    ax.set_xticks(np.arange(0,len(bins)-1,1))
