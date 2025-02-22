# Visualization functions

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind  # For significance testing
import pandas as pd

def plot_participants_per_time(df, time_period='day', title=None):
    """
    Plots the additive distribution of days/weeks per participant for each dataframe in dfs_scaled.

    Parameters:
    dfs_scaled (dict): Dictionary where keys are dataframe names and values are pandas DataFrames.
                       Each DataFrame must have 'participant_id' and either 'day' or 'week' columns.

    Returns:
    None (Displays the plots)
    """
    time = time_period

    # Count unique days/weeks per participant
    participant_time_counts = df.groupby('participant_id')[time].nunique()

    # Count how many participants have at least X days/weeks
    time_distribution = participant_time_counts.value_counts().sort_index()

    # Compute cumulative counts in reverse order
    cumulative_counts = np.cumsum(time_distribution[::-1])[::-1]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(cumulative_counts.index, cumulative_counts.values, color='skyblue')
    plt.xlabel(f'At Least X {time}s per Participant')
    plt.ylabel('Number of Participants')
    if title != None:
        plt.title(f'Additive Distribution of {time}s per Participant for DF: {title}')
    else:
        plt.title(f'Additive Distribution of {time}s per Participant for DF')
    plt.xticks(cumulative_counts.index, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.show()





def plot_var_for_cluster(df_with_cluster, cluster_keys, cluster_var, cluster_label, y_var, y_var_name):
    # Initialize lists to store the means, errors, and labels for the bar chart
    cluster_means = {desc: [] for desc in cluster_keys.keys()}
    cluster_errors = {desc: [] for desc in cluster_keys.keys()}
    labels = []
    p_values = []  # To store p-values from the significance test

    

    # Initialize dictionaries to store data for each cluster
    cluster_data = {desc: df_with_cluster[df_with_cluster[cluster_label] == num] for desc, num in cluster_keys.items()}

    if any(len(cluster_data[desc]) > 0 for desc in cluster_keys.keys()):
        if y_var in df_with_cluster.columns:
            print(f"{y_var} in df")
            # Calculate means and standard errors for each cluster
            for desc, data in cluster_data.items():
                clean_data = data[[y_var, 'num_id']].dropna()
                cluster_values = clean_data[y_var]
                
                cluster_mean = cluster_values.mean()
                cluster_error = cluster_values.sem()  # Standard error of the mean
                
                cluster_means[desc].append(cluster_mean)
                cluster_errors[desc].append(cluster_error)
                
                print(f"{len(clean_data['num_id'].unique())} {desc} {cluster_label} participants")

            # Perform significance test (independent t-test) between all pairs of clusters
            for j in range(len(cluster_keys) - 1):
                for k in range(j + 1, len(cluster_keys)):
                    desc1 = list(cluster_keys.keys())[j]
                    desc2 = list(cluster_keys.keys())[k]
                    t_stat, p_value = ttest_ind(cluster_data[desc1][y_var], cluster_data[desc2][y_var], equal_var=False)
                    p_values.append(p_value)

            labels.append(y_var)

        # Plotting the bar chart
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_keys)))  # Generate colors for each cluster

        rects = []
        for i, (desc, color) in enumerate(zip(cluster_keys.keys(), colors)):
            rect = ax.bar(x + (i - len(cluster_keys) / 2) * width / len(cluster_keys), 
                        cluster_means[desc], width / len(cluster_keys), 
                        yerr=cluster_errors[desc], capsize=5, label=f'{desc.capitalize()} {cluster_label}', color=color)
            rects.append(rect)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('DataFrame')
        ax.set_ylabel(f'{y_var_name}  Mean')
        ax.set_title(f'{y_var_name} Mean by {cluster_label} Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Add significance annotations
        for i, p_value in enumerate(p_values):
            if p_value < 0.05:  # Add annotation only if p-value is significant
                ax.text(x[i % len(labels)], max([cluster_means[desc][i // len(labels)] for desc in cluster_keys.keys()]) + 0.1, '*', ha='center', va='bottom')

        # Attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects):
            for rect in rects:
                for r in rect:
                    height = r.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(r.get_x() + r.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects)

        fig.tight_layout()

        plt.show()
