### Functions and packages for data analysis
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx 
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from fastdtw import dtw
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import DBSCAN
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from fastdtw import dtw
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from outliers import smirnov_grubbs as grubbs



import pandas as pd
import matplotlib.pyplot as plt

def process_cluster_data(scaled_df, 
                         cluster_variable='mobility',
                         required_weeks=None, 
                         plot_histogram=True, 
                         plot_participant_lines=True):
    """
    Process and visualize cluster variable data from a DataFrame.
    
    The function:
      1. Filters the DataFrame to include only the necessary columns.
      2. Groups by participant and week, computing the mean of the specified variable.
      3. Drops any rows with missing values for that variable.
      4. Plots a histogram of participant counts by week.
      5. Pivots the data to create a matrix with participants as rows and weeks as columns.
      6. Filters participants to include only those with complete data for the required weeks.
      7. Optionally plots a line graph of the variable's weekly averages per participant.
    
    Parameters:
      scaled_df (pd.DataFrame): DataFrame containing at least 'num_id', 'week', and the cluster variable.
      cluster_variable (str): The name of the variable to process (default 'mobility').
      required_weeks (list of float/int): Weeks that must be present for each participant. 
                                          Defaults to [1.0, 2.0, 3.0, 4.0] if not provided.
      plot_histogram (bool): Whether to plot the histogram of participant counts by week.
      plot_participant_lines (bool): Whether to plot individual participant lines over weeks.
      
    Returns:
      filtered_df (pd.DataFrame): DataFrame of participants with complete data for the required weeks.
    """
    # Set default required weeks if none provided
    if required_weeks is None:
        required_weeks = [1.0, 2.0, 3.0, 4.0]
    
    # Step 1: Filter DataFrame to just include 'num_id', 'week', and the cluster variable
    var_df = scaled_df[['num_id', 'week', cluster_variable]]
    print('Var df:\n', var_df.head())
    
    # Step 2: Group by participant and week and compute the average of the cluster variable
    grouped_df = var_df.groupby(['num_id', 'week'], as_index=False)[cluster_variable].mean()

    # Step 3: Drop rows (participant/week) with NA values for the cluster variable
    cleaned_df = grouped_df.dropna(subset=[cluster_variable])
    
    # Step 4: Count the number of participants for each unique week
    week_counts = cleaned_df['week'].value_counts().sort_index()
    
    # Step 5: Plot the histogram of participants per week if requested
    if plot_histogram:
        plt.figure(figsize=(10, 6))
        plt.bar(week_counts.index, week_counts.values, color='skyblue')
        plt.xlabel('Week')
        plt.ylabel('Number of Participants')
        plt.title(f'Distribution of Participants by Week for Variable: {cluster_variable}')
        plt.xticks(week_counts.index)
        plt.show()

    
    # Step 6: Pivot the DataFrame to create a matrix with participants as rows and weeks as columns
    # fill na with 0 then mask out later
    pivot_df = grouped_df.pivot(index='num_id', columns='week', values=cluster_variable).reset_index().fillna(0)
    # Step 4: Remove the columns name ('week') for clarity, since the index is not actually the weeks 
    pivot_df.columns.name = None

    # Step 7: Identify the columns corresponding to the required weeks
    week_columns = [week for week in required_weeks if week in pivot_df.columns]
    
    
    # Step 9: Print the shape of the filtered DataFrame
    pivot_df_for_graphing = pivot_df.mask(pivot_df==0)
    
    # Step 10: Plot the variable's scores for each participant over the required weeks if requested
    if plot_participant_lines:
        plt.figure(figsize=(12, 8))
        for _, row in pivot_df_for_graphing.iterrows():
            plt.plot(week_columns, row[week_columns], marker='o', label=f'Participant {row["num_id"]}')
        plt.xlabel('Week')
        plt.ylabel(cluster_variable)
        plt.title(f'{cluster_variable} Avg Scores Each Week for Each Participant\n'
                  f'(Each colored line represents a unique participant) \n'
                  f'{pivot_df.shape[0]} Participants')
        plt.show()
    
    return pivot_df



def compute_distance_matrix(filtered_df, required_weeks, verbose=False):
    """
    Compute a symmetric euclidean distance matrix for the data in filtered_df.
    Only the columns in required_weeks are used.
    """
    # Select the relevant columns and ensure they are numeric
    var_weeks = filtered_df[list(required_weeks)]
    n = len(var_weeks)
    dtw_matrix = np.zeros((n, n))
    
    # Compute pairwise DTW distances
    for i in range(n):
        series_i = var_weeks.iloc[i].to_numpy()
        for j in range(i + 1, n):
            series_j = var_weeks.iloc[j].to_numpy()
            # Compute DTW distance (if dtw returns a tuple, take the first element)
            d = np.linalg.norm(series_i - series_j)
            if isinstance(d, (tuple, list)):
                d = d[0]
            dtw_matrix[i, j] = d
            dtw_matrix[j, i] = d  # symmetry
    if verbose:
        print("Euclidean distance matrix stats:")
        print("Min:", np.min(dtw_matrix))
        print("Max:", np.max(dtw_matrix))
        print("Mean:", np.mean(dtw_matrix))

    return dtw_matrix

def compute_dtw_matrix(filtered_df, required_weeks, verbose=False):
    """
    Compute a symmetric DTW distance matrix for the data in filtered_df.
    Only the columns in required_weeks are used.
    """
    # Select the relevant columns and ensure they are numeric
    var_weeks = filtered_df[list(required_weeks)]
    n = len(var_weeks)
    dtw_matrix = np.zeros((n, n))
    
    # Compute pairwise DTW distances
    for i in range(n):
        series_i = var_weeks.iloc[i].to_numpy()
        for j in range(i + 1, n):
            series_j = var_weeks.iloc[j].to_numpy()
            # Compute DTW distance (if dtw returns a tuple, take the first element)
            d = dtw(series_i, series_j)
            if isinstance(d, (tuple, list)):
                d = d[0]
            dtw_matrix[i, j] = d
            dtw_matrix[j, i] = d  # symmetry

    if verbose:
        print("DTW distance matrix stats:")
        print("Min:", np.min(dtw_matrix))
        print("Max:", np.max(dtw_matrix))
        print("Mean:", np.mean(dtw_matrix))

    return dtw_matrix

def cluster_distance_analysis(df, cols, cluster_var, n_clusters_range=range(1, 11), metric='euclidean', verbose=False):
    """
    Compute the DTW distance matrix for filtered_df (using cols),
    embed the distances into Euclidean space with MDS,
    and then evaluate clusters with KMeans using inertia (elbow method),
    silhouette score, and Davies–Bouldin index.
    
    Parameters:
      df: DataFrame with rows corresponding to participants and columns for each week.
      cols: list of columns (numeric) to use for distance matrix vectors.
      cluster_variable: name of the variable (for plot titles).
      n_clusters_range: range of cluster counts to evaluate.
      
    Returns:
      embedding: 2D coordinates from MDS.
      results: dictionary containing inertia, silhouette scores, and DB scores.
    """

    # Step 1: Compute the DTW distance matrix
    if metric=='euclidean':
        distance_matrix = compute_distance_matrix(df, cols, verbose)
    elif metric=='dtw':
        distance_matrix = compute_dtw_matrix(df, cols, verbose)
    else:
        return ValueError
    
    # Check if the distance matrix is valid
    if np.isnan(distance_matrix).any():
        raise ValueError("The computed distance matrix contains NaN values.")
    
    # # Step 2: Use MDS to embed the distance matrix into 2D Euclidean space
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    embedding = mds.fit_transform(distance_matrix)
    
    # Prepare to store evaluation metrics
    inertia_vals = []
    silhouette_scores = []
    db_scores = []
    
    # Evaluate for each number of clusters (note: silhouette & DB indices require at least 2 clusters)
    for n_clusters in n_clusters_range:
        # Run KMeans on the distance matrix
        kmeans = KMeans(n_clusters=n_clusters, random_state=34, n_init=10)
        labels = kmeans.fit_predict(embedding)
        
        # For inertia, we can use all cluster numbers including 1.
        inertia_vals.append(kmeans.inertia_)
        
        # For silhouette and Davies-Bouldin, only compute if n_clusters >= 2
        if n_clusters >= 2:
            labels = kmeans.labels_
            sil_score = silhouette_score(embedding, labels)
            silhouette_scores.append(sil_score)
            
            db_index = davies_bouldin_score(embedding, labels)
            db_scores.append(db_index)
    
    # Plot silhouette scores (starting at 2 clusters)
    plt.figure(figsize=(5, 3))
    plt.plot(list(n_clusters_range)[1:], silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score For Optimal k for {cluster_var}, metric: {metric}\n(Choose highest score)')
    plt.grid()
    plt.show()
    
    # Plot Davies-Bouldin Index (starting at 2 clusters)
    plt.figure(figsize=(5, 3))
    plt.plot(list(n_clusters_range)[1:], db_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title(f'Davies-Bouldin Index For Optimal k for {cluster_var}, metric: {metric}\n(Choose lowest score)')
    plt.grid()
    plt.show()
    
    # Return the embedding and metrics in case you need them further
    results = {
        "inertia": inertia_vals,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": db_scores
    }
    return distance_matrix, embedding, results





def cluster_dtw_analysis(df, cols, eps_values=np.linspace(0.05, 1.0, 20), min_samples=10, verbose=False):
    """
    Compute the DTW distance matrix for filtered_df (using required_weeks),
    and then evaluate clusters with silhouette score, then embed the distances 
    into Euclidean space with MDS and evaluate with Davies–Bouldin index
    
    Parameters:
      df: DataFrame with rows corresponding to participants and columns for each week.
      cols: list of week identifiers (columns) to use for dtw matrix
      cluster_variable: name of the variable (for plot titles).

      
    Returns:
      dtw_matrix: pandas df of DTW distance matrix for <cols> by <cols>
      embedding: 2D coordinates from MDS.
      results: dictionary containing silhouette scores and DB scores.
    """
    # Step 1: Compute the DTW distance matrix
    dtw_matrix = compute_dtw_matrix(df, cols, verbose)

    # Check if the distance matrix is valid
    if np.isnan(dtw_matrix).any():
        raise ValueError("The computed dtw_matrix contains NaN values.")

    # # Step 2: Use MDS to embed the distance matrix into 2D Euclidean space
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    embedding = mds.fit_transform(dtw_matrix)
    
    sil_scores = []
    num_clusters = []
    noise_ratios = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(dtw_matrix)
        
        # Count clusters (excluding noise)
        unique_labels = set(labels) - {-1}
        if verbose:
            print(f"eps: {eps:.2f}, clusters: {len(unique_labels)}, noise: {np.sum(labels == -1)}")
        num_clusters.append(len(unique_labels))
        
        # Compute noise ratio
        noise_ratio = np.sum(labels == -1) / len(labels)
        noise_ratios.append(noise_ratio)
        
        # Compute clustering metrics (only if there are at least 2 clusters)
        if len(unique_labels) > 1:
            non_noise_idx = labels != -1
            sil_scores.append(
                silhouette_score(
                    dtw_matrix[np.ix_(non_noise_idx, non_noise_idx)],
                    labels[non_noise_idx],
                    metric='precomputed'
                )
            )
        else:
            sil_scores.append(0)

    # Convert None to NaN for plotting
    sil_scores = np.array(sil_scores)
    eps_values = np.array(eps_values)

    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epsilon (eps)")
    ax1.set_ylabel("Silhouette Score (select highest score)")
    ax1.plot(eps_values, sil_scores, 'bo-', label="Silhouette Score")
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of Clusters / Noise Ratio")
    ax2.plot(eps_values, num_clusters, 'go-', label="Number of Clusters")
    ax2.plot(eps_values, noise_ratios, 'mo-', label="Noise Ratio")
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper right")

    plt.title("DBSCAN Performance Over Different Eps Values")
    plt.show()

    results = {
        "silhouette_scores": sil_scores
    }

    return dtw_matrix, embedding, results




def normalize_df(df, columns_to_scale):
    scaler = MinMaxScaler()
    # Select all columns which aren't in columns_to_scale
    non_numeric_cols = list(set(df.columns.to_list()).difference(columns_to_scale))
    scaled_df = df[non_numeric_cols]
    for x_col in columns_to_scale:
        #print('\nFOR COLUMN:', x_col)
        col_df = df[['num_id', x_col]].dropna()  # Drop NaNs
        col_df_clean = col_df[col_df[x_col] != 0]  # Remove 0 values

        # Reshape to 2D array as scaler expects
        if col_df_clean[x_col].shape[0] > 1:
            col_df_clean[f'{x_col}_scaled2'] = scaler.fit_transform(col_df_clean[[x_col]])

            ## Find outliers using Smirnov_Grubbs test and flatten the result
            non_outliers = grubbs.test(col_df_clean[[f'{x_col}_scaled2']].to_numpy(), alpha=0.05).flatten()
            # Only keep numbers in x_col_scaled which are not-outliers
            col_df_clean[f'{x_col}_scaled'] = col_df_clean[f'{x_col}_scaled2'].where(col_df_clean[f'{x_col}_scaled2'].isin(non_outliers))

            # Assign scaled values back to the original DataFrame, aligning indices
            scaled_df[x_col] = pd.Series(
                col_df_clean[f'{x_col}_scaled'].values,
                index=col_df_clean.index
            )
    return scaled_df



def hierarchical_agg_plot(condensed_matrix):
    # Handle non-finite values
    condensed_matrix = np.where(np.isinf(condensed_matrix), np.nan, condensed_matrix)
    
    # Step 3: Create clustering with different solutions
    Z1 = linkage(condensed_matrix, method='single', metric='euclidean')
    Z2 = linkage(condensed_matrix, method='complete', metric='euclidean')
    Z3 = linkage(condensed_matrix, method='average', metric='euclidean')
    Z4 = linkage(condensed_matrix, method='ward', metric='euclidean')

    # Step 4: Plot the dendrograms
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1), dendrogram(Z1), plt.title('Single')
    plt.subplot(2,2,2), dendrogram(Z2), plt.title('Complete')
    plt.subplot(2,2,3), dendrogram(Z3), plt.title('Average')
    plt.subplot(2,2,4), dendrogram(Z4), plt.title('Ward')
    plt.show()
