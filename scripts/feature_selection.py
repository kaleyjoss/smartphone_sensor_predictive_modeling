## Feature Selection

############ LOAD in PACKAGES  #############
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.linear_model import LogisticRegression
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

def randomized_logistic_regression(X, y, num_samples=200, selection_threshold=0.1):
    """
    Performs Randomized Logistic Regression (RLR) for feature selection.

    Parameters:
    - X: np.array or DataFrame, feature matrix
    - y: np.array, target vector (binary classification)
    - num_samples: int, number of bootstrap samples
    - selection_threshold: float, threshold for feature importance selection

    Returns:
    - selected_features: List of selected feature indices
    - mean_coefs: Array of mean absolute coefficients for all features
    """
    n_features = X.shape[1]
    coef_matrix = np.zeros((num_samples, n_features))

    for i in range(num_samples):
        # Randomly sample 70% of data with replacement
        idx = np.random.choice(X.shape[0], size=int(0.7 * X.shape[0]), replace=True)
        X_sample, y_sample = X.iloc[idx], y.iloc[idx]

        # Fit L1-penalized Logistic Regression
        model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100)
        model.fit(X_sample, y_sample)

        # Store the absolute coefficients
        coef_matrix[i, :] = np.abs(model.coef_)

    # Compute mean absolute coefficients across all bootstrap runs
    mean_coefs = np.mean(coef_matrix, axis=0)

    # Select features where mean coefficient is above the threshold
    selected_features = np.where(mean_coefs > selection_threshold)[0]

    # Get the corresponding feature names
    selected_feature_names = X.columns[selected_features]
    

    return selected_feature_names, mean_coefs



############# RUN A hierarchical agg clustering on averaging all VARS, V1 ##############
def create_condensed_matrix(data, scaled_x_cols, scaled_y_cols):
    full_df = data[['participant_id', 'week', 'day', 'dt'] + scaled_x_cols + scaled_y_cols]
    full_df = full_df.dropna()
    #full_df_v1 = full_df_v1.drop(columns='aggregate_communication_scaled')
    print('In full_dt_v1 there are', len(full_df['participant_id'].unique()), 'subjects')

    for col in scaled_x_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Create hierarchical clustering of all the variables

    # # Group by 'participant_id' and calculate the mean for each variable
    keep_columns = ['participant_id'] + [var for var in full_df.columns.to_list() if var.endswith('_scaled_int') or 'phq2_sum' in var]
    avg_df = full_df[keep_columns].groupby('participant_id').mean().reset_index()

    # Delete non-numerical/id rows (participant id)
    data = avg_df.iloc[0:, 1:] 

    # Handle non-finite values
    data = data.replace([float('inf'), -float('inf')], pd.NA)  # Replace inf/-inf with NaN
    print('data.shape',data.shape)
    # Step 1: Convert Pearson correlation matrix to distance matrix
    data_corr = data.corr()

    # Turn corr matrix into a distance matrix
    distance_matrix = 1 - data_corr
    print('distance_matrix.shape',distance_matrix.shape)

    # Step 2: Convert distance matrix to condensed distance matrix
    # pdist expects a square distance matrix, so use squareform to validate later if needed.
    condensed_matrix = squareform(distance_matrix, checks=False)
    print('condensed_matrix.shape',condensed_matrix.shape)

    return distance_matrix, condensed_matrix, data


def flatten_matrix(corr_matrix):
    indices = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i, corr_matrix.shape[1]):
            if not i==j:
                indices.append([i,j])
    return [corr_matrix.iloc[row, col] for row, col in indices]

import pandas as pd

def zero_matrix(corr_matrix):
    # Initialize a new matrix filled with zeros
    masked_matrix = pd.DataFrame(0.0, index=corr_matrix.index, columns=corr_matrix.columns)
    
    # Iterate through the matrix to get the index pairs
    indices = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i, corr_matrix.shape[1]):
            if not i == j:
                indices.append([i, j])
                # Copy the corresponding value from corr_matrix to masked_matrix
                masked_matrix.iloc[i, j] = corr_matrix.iloc[i, j]

    return masked_matrix



def upper_triangle(corr_matrix):
    indices = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i, corr_matrix.shape[1]):
            if not i==j:
                indices.append([i,j])
    return indices




def average_matrix(symptom_matrices):
    '''symptom_matrices = dict 
        where keys are subjects
        and each subject key has a correlation matrix
        of the PMC of numerical variables

        returns average_matrix, a 2D array which is a 
        average of all the subject correlation matrices
    '''
    # Extract matrices
    matrices = list(symptom_matrices.values())
    # Ensure all matrices have the same shape
    if all(len(m) == len(matrices[0]) for m in matrices):
        # Stack matrices along a new axis
        stacked_matrices = np.stack(matrices, axis=0)
        # Compute the average matrix
        average_matrix = np.mean(stacked_matrices, axis=0)
        #print("Average Correlation Matrix:\n", average_matrix)
        # Transform to distance matrix
        distance_matrix = 1 - average_matrix
        # Compute pairwise distances
        condensed_matrix = squareform(distance_matrix, checks=False)
        
        return distance_matrix

    else:
        print("All matrices must have the same dimensions.")



def make_symptom_matrices(df, ignore_cols, num_to_plot=0):
    symptom_matrices = {}
    flattened_matrices = {}

    subs_sm = []
    subs_fm = []
    print(f'In df there are {len(df['num_id'].unique())} subjects.')
    count=0
    for sub in df['num_id'].unique():
        data = df[df['num_id']==sub] # filter for each specific sub
        # keep only numerical/changing columns
        keep_columns = list(set(df.columns.to_list()) - set(ignore_cols))
        data = data[keep_columns] 
        # transform into correlation matrix
        correlation_matrix = data.corr() 
        # Replace inf with 1
        correlation_matrix = correlation_matrix.replace([float('inf')], 1)  
        # Replace -inf with -1
        correlation_matrix = correlation_matrix.replace([-float('inf')], -1)  
        # Replace NaN with 0
        correlation_matrix = correlation_matrix.replace([np.nan], 0)  
        # add subs with non-empty matrices to subs_sm
        if not correlation_matrix.empty and not ((correlation_matrix == 0).all().all()):
            subs_sm.append(sub) 
            # add entire corr matrix to list
            symptom_matrices[sub] = correlation_matrix
            # extract unique values from upper triangle into vector
            vector = flatten_matrix(correlation_matrix) 
            # add nonzero vector to list
            if not len(vector)==0:
                flattened_matrices[sub] = vector
                subs_fm.append(sub)
            
            if num_to_plot>0:
                if count<num_to_plot:
                    # # Heatmap
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.1f', linewidths=0.5)
                    plt.title(f"Subject {sub}: Correlation Matrix")
                    plt.show()  

                    # Line/Scatter plot
                    x = np.arange(len(vector))
                    plt.figure(figsize=(5, 3))
                    plt.title(f"Subject {sub}: Plot of condensed vector (flattened matrix)")
                    plt.scatter(x, vector, label="Points", color="blue")
                    plt.show() 

                    count+=1 

    print(len(symptom_matrices.keys()), 'subs with symptom matrices')
    print(len(flattened_matrices.keys()), 'filled condensed arrays')

    return symptom_matrices, flattened_matrices



############# Plot Hierarchical Agglomerative Clustering ##################
def plot_hier_agg(flattened_matrices, matrix_labels=None, is_dict=False, group_title=None):
    # Compute pairwise distances
    if is_dict is True:
        distances = pdist(list(flattened_matrices.values()), metric='euclidean')
    else:
        distances = pdist(flattened_matrices)

    # Perform hierarchical clustering 

    # Step 3: Create clustering with different solutions
    Z1 = linkage(distances, method='single', metric='euclidean')
    Z2 = linkage(distances, method='complete', metric='euclidean')
    Z3 = linkage(distances, method='average', metric='euclidean')
    Z4 = linkage(distances, method='ward', metric='euclidean')


    # Step 4: Plot the dendrograms
    plt.figure(figsize=(10, 8))
    if not (matrix_labels is None):
        labels = matrix_labels
    elif is_dict is True:
        labels = list(flattened_matrices.keys())
    else:
        labels=None

    print(group_title)
    if is_dict is True:
        print(len(flattened_matrices.keys()), 'matrices included')
    else:
        print(flattened_matrices.shape, 'variables included')
    plt.subplot(2,2,1), dendrogram(Z1, labels=labels, leaf_rotation=90, leaf_font_size=8), plt.title('Single')
    plt.subplot(2,2,2), dendrogram(Z2, labels=labels, leaf_rotation=90, leaf_font_size=8), plt.title('Complete')
    plt.subplot(2,2,3), dendrogram(Z3, labels=labels, leaf_rotation=90, leaf_font_size=8), plt.title('Average')
    plt.subplot(2,2,4), dendrogram(Z4, labels=labels, leaf_rotation=90, leaf_font_size=8), plt.title('Ward')
    plt.tight_layout()  # Prevent overlapping elements
    plt.show()


############# Create Hierarchical Agglomerative Clustering Labels ##################
def hier_agg_clustering(matrices, labels, n_clusters=2, linkage='ward', is_dict=False):
    hierarchical_cluster = AgglomerativeClustering(
        n_clusters=n_clusters,  # Number of clusters
        linkage=linkage  # Linkage method
    )

    if is_dict is True:
        cluster_labels = hierarchical_cluster.fit_predict(list(matrices.values()))
    else:
        cluster_labels = hierarchical_cluster.fit_predict(matrices)
    
    # # Map subject ids to labels
    subject_clusters = dict(zip(labels, cluster_labels))

    # Print cluster counts
    # cluster_counts = pd.Series(list(subject_clusters.values())).value_counts()
    # print(cluster_counts)

    return subject_clusters


############# Plot Individual Networks ##################
def plot_network(df, corr_matrix_rows=None, title=None, threshold=0.5, scale_weights=False, fixed_positions=None, draw_edge_weights=False):
    if corr_matrix_rows:
        df_corr = df[corr_matrix_rows]
    else:
        df_corr = df
    # Create correlation matrix from df
    corr_matrix = df_corr.corr()
    corr_matrix = zero_matrix(corr_matrix)
    #display(corr_matrix)
    # If correlation matrix has nonzero values
    if not (corr_matrix==0).all().all():
        # Create a graph
        G = nx.Graph()
        # Add edges based on a correlation threshold
        for i, row in corr_matrix.iterrows():
                G.add_node(i)
                for j, value in row.items():
                    G.add_node(j)
                    if isinstance(value, (int, float)) and i != j and abs(value) > threshold: # Avoid self-loops and weak correlations
                        G.add_edge(i, j, weight=value*5)
            
        # Extract edge weights and normalize them for width scaling
        weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
        if scale_weights==True:
            min_width, max_width = 1, 10  # Define thickness range
            if (max(weights) - min(weights))==0:
                scaled_weights = [5 for _ in weights]
            else:
                scaled_weights = [(w - min(weights)) / (max(weights) - min(weights)) * (max_width - min_width) + min_width for w in weights]
        else:
            scaled_weights=weights
        # Set edge colors: blue for negative, red for positive 
        edge_colors = ['red' if d['weight']<0 else 'green' for (u, v, d) in G.edges(data=True)]
        if fixed_positions is not None:
            missing_nodes = [n for n in G.nodes if n not in fixed_positions]
            if len(missing_nodes)>0:
                print("Missing nodes:", missing_nodes)
            pos = fixed_positions # Use the fixed positions for the layout
        else:
            pos = nx.spring_layout(G, seed=42)  # Layout for better visualization

        # Size of figures
        plt.figure(figsize=(3, 2))
        
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=20,
            font_size=7,
            width=scaled_weights,  # Set edge thickness
            edge_color=edge_colors,
            font_weight="bold"
        )
        if draw_edge_weights == True:
            # Draw edge labels (correlation values)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}
            )
        if title != None:
            plt.title(title)
        plt.tight_layout()
        plt.show()



############## PCA on 1st component of all clusters #############

def pca_on_clusters(df, cluster_dict, n_clusters, n_components=1):
    pca_dict = {}
    all_scores = []  # Store PCA-transformed data across clusters

    for i in range(n_clusters):
        keep_columns = [var for var in df.columns.to_list() if cluster_dict.get(var) == i]
        if not keep_columns:
            print(f"Skipping cluster {i}: No variables assigned.")
            continue

        print(f'Cols for cluster {i}: {keep_columns}')
        df = df.dropna()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        c1_pca = pca.fit_transform(df[keep_columns])

        #print(f'Explained variance of PC1 for cluster {i}: {pca.explained_variance_ratio_}')

        # PCA Loadings (not used further in this function)
        loadings = pd.DataFrame(pca.components_, columns=df[keep_columns].columns,
                                index=[f'PC{x+1}' for x in range(pca.n_components_)])

        # # Heatmap of the loadings
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(loadings, annot=False, cmap='coolwarm', center=0)
        # plt.title(f'PCA Component Loadings for V2 Option B: Taking out communication vars\n{len(full_df_v2['participant_id'].unique())} Subjects\nExplained variance of each PC: {pca.explained_variance_ratio_}')
        # plt.xlabel('Original Features')
        # plt.ylabel('Principal Components')
        # plt.show()


        # PCA Scores (transformed data)
        c1_scores_df = pd.DataFrame(c1_pca, columns=[f'c{i}_PC{x+1}' for x in range(pca.n_components_)])
        c1_scores_df[['participant_id','dt','week']] = df[['participant_id','dt','week']]
        
        # Store in dictionary
        pca_dict[i] = {
            'id': i,
            'columns': keep_columns,
            'explained_variance': pca.explained_variance_ratio_,
            'df': c1_scores_df
        }
        
        # Add PCA scores column to list
        all_scores.append(c1_scores_df)

    # Concatenate PCA scores across clusters
    combined_pca_df = pd.concat(all_scores, axis=1)

    return combined_pca_df, pca_dict





def merge_df_via_cluster_pca_dict(df, pca_dict, on_columns):
# Merge all the pc cluster loadings onto the original df
    for cluster in pca_dict.keys():
        df = df.merge(pca_dict[cluster]['df'], on=['participant_id','week','dt'])
        df = df.rename(columns={f"c{cluster}_PC1": pca_dict[cluster]['name']}) 
    return df

