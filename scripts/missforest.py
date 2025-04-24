
################## Impute using MissForest #########################

def simulate_missing_data(original_df, missing_percentage=0.1, random_state=None):
    np.random.seed(random_state)

    total_cells = original_df.shape[0] * original_df.shape[1]
    cells_to_impute = int(total_cells * missing_percentage)
    df_mask_out = original_df.copy()
    #print(f"Masking out 0.1 of {total_cells} ({cells_to_impute}), out of full df {df_mask_out.shape}")

    # Generate all possible (row, col) pairs
    all_indices = [(i, j) for i in range(df_mask_out.shape[0]) for j in range(df_mask_out.shape[1])]

    # Randomly choose unique pairs
    selected_indices = np.random.choice(len(all_indices), size=cells_to_impute, replace=False)
    masked_positions = [all_indices[idx] for idx in selected_indices]

    masked_values = []

    for row, col in masked_positions:
        value = df_mask_out.iat[row, col]
        while pd.isna(value) and (row + 1 < df_mask_out.shape[0]) and (col + 1 < df_mask_out.shape[1]):
            row += 1
            value = df_mask_out.iat[row, col]
            col += 1
            value = df_mask_out.iat[row, col]
        masked_values.append((row, col, value))
        df_mask_out.iat[row, col] = np.nan

    df_mask_in = pd.DataFrame(masked_values, columns=['row', 'column', 'value'])

    return df_mask_out, df_mask_in




def simulate_missing_data_bycol(original_df, col, missing_percentage=0.1, random_state=None):
    col_index = original_df.columns.get_loc(col)

    total_cells = len(original_df)
    cells_to_impute = int(total_cells * missing_percentage)
    print(f"Masking out 0.1 of {total_cells} ({cells_to_impute}), for col {col} at index {col_index}")

    # Generate all possible row numbers 
    all_indices = [i for i in range(total_cells)]

    # Randomly choose unique rows
    np.random.seed(random_state)
    selected_indices = np.random.choice(len(all_indices), size=cells_to_impute, replace=False)
    masked_positions = [all_indices[idx] for idx in selected_indices]
    masked_values = []

    df_mask_out = original_df[col].copy()
    print('df_mask_out shape', df_mask_out.shape)
    for row in masked_positions:
        value = df_mask_out.iat[row]
        while pd.isna(value) and (row + 1 < df_mask_out.shape[0]):
            row += 1
            value = df_mask_out.iat[row]
        while pd.isna(value) and (row + 1 < df_mask_out.shape[0]):
            row -= 1
            value = df_mask_out.iat[row]
        masked_values.append((row, col_index, value))
        df_mask_out.iat[row] = np.nan

    df_mask_in = pd.DataFrame(masked_values, columns=['row', 'column', 'value'])

    return df_mask_out, df_mask_in




# Updated compute_imputation_error provided above
def compute_imputation_error(original_df, imputed_df, df_mask_in):
    """
    Compute the mean squared error between original and imputed values,
    but only for those entries where missingness was simulated (according to mask).
    """
    # Make sure the dfs have the same columns/size
    different_columns = len(set(original_df.columns) - set(imputed_df.columns))
    if different_columns > 0 or (original_df.shape[0] != imputed_df.shape[0]) or (original_df.shape[1] != imputed_df.shape[1]):
        print(f"ERROR: Columns not same: {different_columns} or ERROR: Not same size. original_df {original_df.shape}, imputed_df {imputed_df.shape}")
        raise ValueError 
    
    else:
        original_values = df_mask_in['value'].to_list()
        imputed_values = []
        for i, i_row in df_mask_in.iterrows():
            row = int(i_row['row'])
            col = int(i_row['column'])
            imputed_value = imputed_df.iat[row, col]
            imputed_values.append(imputed_value)


        # Evaluate MSE & R-squared
        print("NaNs in original_values:", np.isnan(original_values).sum(), original_values)
        print("NaNs in imputed_values:", np.isnan(imputed_values).sum(), imputed_values)        
        if np.isnan(original_values).sum() > 0 or  np.isnan(imputed_values).sum() > 0:
            return None, None, None, None
        rmse_value = rmse(original_values, imputed_values)
        rmse_scaled = rmse_value / np.mean(original_values)

        mean_baseline = [np.mean(original_values)] * len(original_values)

        base_r2 = sklearn.metrics.r2_score(original_values, mean_baseline)
        imp_r2 = sklearn.metrics.r2_score(original_values,imputed_values)
        

        print(f"RMSE (as percentage of value) is {rmse_scaled}")
        print(f"Base R² is {base_r2}, Imputed R² is {imp_r2}")

        if base_r2 > imp_r2:
            print(f"Using Base R²")
            imputed_values = mean_baseline
            imp_r2 = base_r2
        else:
            print(f"Using Imp R²")
        
        return original_values, imputed_values, rmse_scaled, imp_r2

def missforest_imputation_bysub(original_df, cols_to_impute, imputation_threshold=0.6, error_threshold=0.8, verbose=False):
    """
    Impute missing values using MissForest for each subject (identified by 'num_id')
    if the subject has at least imputation_threshold proportion of non-missing data.
    """
    r2_list = {}
    imputed_sub_dfs = []
    imputed_subs = []
    nonimputed_subs = []

    for sub in original_df['num_id'].unique():
        sub_og = original_df[original_df['num_id'] == sub].copy()
        if verbose:
            print(f'For sub {sub}, {sub_og.shape[0]} rows, {sub_og.shape[0]*sub_og.shape[1]} cells, {sub_og.isna().sum().sum()}')
        
        # Only impute for subjects with 25 days or more
        if sub_og.shape[0] < 25:
            if verbose:
                print(f"Skipping subject {sub}: all imputation columns are missing.\n\n")
            imputed_sub_dfs.append(sub_og)
            nonimputed_subs.append(sub)
            continue
        
        else:
            nonimpute_cols = [col for col in sub_og.columns if col not in cols_to_impute]
            sub_og = sub_og[cols_to_impute].copy()
            
            # Drop columns that are entirely NaN
            fully_missing_cols = sub_og.columns[sub_og.isna().all()]
            if len(fully_missing_cols) > 0:
                sub_og_clean = sub_og.drop(columns=fully_missing_cols)
                nonimpute_cols = nonimpute_cols + list(fully_missing_cols)
                # Skip this subject if all columns were removed
                if sub_og_clean.shape[1] == 0:
                    if verbose:
                        print(f"Skipping subject {sub}: all imputation columns are missing.\n\n")
                    imputed_sub_dfs.append(sub_og)
                    nonimputed_subs.append(sub)
                    continue
            else:
                sub_og_clean = sub_og

            # Calculate the proportion of non-missing data in impute_df_clean
            non_missing_proportion = sub_og_clean.notna().mean().mean()
            
            # Check that it's over the imputation threshold (ie 60%, 70% etc)
            if non_missing_proportion >= imputation_threshold:
                df_mask_out, df_mask_in = simulate_missing_data(sub_og_clean, missing_percentage=0.1, random_state=None)
                
                # # Drop any columns in df_sim_missing that became fully NaN
                # fully_missing_cols = df_sim_missing.columns[df_sim_missing.isna().all()]
                # if len(fully_missing_cols) > 0:
                #     impute_df_clean = df_sim_missing.drop(columns=fully_missing_cols)
                #     sim_missing_mask = pd.DataFrame(sim_missing_mask, index=df_sim_missing.index, 
                #                                     columns=df_sim_missing.columns).drop(columns=fully_missing_cols).to_numpy()
                #     nonimpute_cols = nonimpute_cols + list(fully_missing_cols)
                
                # Check if all the imputation columns are empty and skip if so
                if df_mask_out.isna().sum().sum() == (df_mask_out.shape[0] * df_mask_out.shape[1]):
                    if verbose:
                        print(f"Skipping subject {sub}: all imputation columns are missing.\n\n")
                    imputed_sub_dfs.append(sub_og)
                    nonimputed_subs.append(sub)
                    continue
                else:
                    # Otherwise, impute the data
                    imputer = MissForest()
                    sub_imputed = imputer.fit_transform(df_mask_out)
                    
                    original_values, imputed_values, rmse_scaled, r2 = compute_imputation_error(sub_og_clean, sub_imputed, df_mask_in)
                    if np.isnan(original_values).any() or np.isnan(imputed_values).any():
                        print(f"Skipping sub, original values couldn't have enough \n\n")
                        imputed_sub_dfs.append(sub_og)
                        nonimputed_subs.append(sub)
                    # If the r-squared of imputed data is over 0.75, keep the imputation
                    if r2 > error_threshold:
                        if verbose:
                            print(f"Using imputed values for {sub}: RMSE is {rmse_scaled},  R² is {r2}\n\n")
                            
                        sub_df_nonimputed = sub_og[nonimpute_cols]
                        sub_imputed = sub_df_nonimputed.merge(sub_imputed, on='day', left_index=True, right_index=True)
                        imputed_sub_dfs.append(sub_imputed)
                        imputed_subs.append(sub)
                        r2_list[sub] = r2
                    # Otherwise, don't impute that sub
                    else:
                        if verbose: 
                            print(f"Skipping sub {sub}, R-squared of imputation was below {error_threshold}\n\n")
                        imputed_sub_dfs.append(sub_og)
                        nonimputed_subs.append(sub)
                    

                    
            else:
                if verbose:
                    print(f"Skipping subject {sub}: Not enough non-missing data ({non_missing_proportion:.2f}).\n\n")
                imputed_sub_dfs.append(sub_og)
                nonimputed_subs.append(sub)

    imputed_df = pd.concat(imputed_sub_dfs)
    return imputed_df, r2_list, imputed_subs, nonimputed_subs

def missforest_imputation_bycol(original_df, cols_to_impute, imputation_threshold=0.3, error_threshold=0.8, verbose=False):
    """
    Impute missing values using MissForest for each subject (identified by 'num_id')
    if the subject has at less than imputation_threshold proportion of missing data.
    """
    r2_list = {}
    imputed_cols = []
    nonimputed_cols = []


    nonimputed_cols = [col for col in original_df.columns if col not in cols_to_impute]
    nonimputed_df = original_df[nonimputed_cols]
    imputed_df = pd.DataFrame()
    df = original_df[cols_to_impute].copy()
    
    # Drop columns that are entirely NaN
    fully_missing_cols = df.columns[df.isna().all()]
    if len(fully_missing_cols) > 0:
        df_clean = df.drop(columns=fully_missing_cols)
        cols_to_impute.remove(fully_missing_cols)
        nonimputed_df = pd.concat([nonimputed_df, df_clean[fully_missing_cols]], axis=1)
    else:
        df_clean = df

    for col in cols_to_impute:
        if verbose:
            print(f"Attempting to impute {col}...")
        # Calculate the proportion of missing data in df_clean
        missing_perc = df_clean[col].isna().sum() / (df_clean.shape[0])
        
        # Check that it's over the imputation threshold (ie 60%, 70% etc)
        if not missing_perc < imputation_threshold:
            if verbose:
                print(f"Not imputing: missing perc is {missing_perc}")
        else:
            if verbose:
                print(f"Imputing: missing perc is {missing_perc}...")

            df_mask_out, df_mask_in = simulate_missing_data_bycol(df_clean, col, missing_percentage=0.1, random_state=None)
            df_mask_out = pd.DataFrame(df_mask_out)
            # # Drop any columns in df_mask_out that became fully NaN
            # fully_missing_cols = df_mask_out.columns[df_mask_out.isna().all()]
            # if len(fully_missing_cols) > 0:
            #     df_mask_out = df_mask_out.drop(columns=fully_missing_cols)
            #     df_mask_in = pd.DataFrame(df_mask_in, index=df_mask_out.index, 
            #                                     columns=df_mask_out.columns).drop(columns=fully_missing_cols).to_numpy()
            #     nonimpute_cols = nonimpute_cols + list(fully_missing_cols)
            
            # Otherwise, impute the data
            imputer = MissForest()
            imputed_col = imputer.fit_transform(df_mask_out)
            
            original_values, imputed_values, rmse_scaled, r2 = compute_imputation_error(df_clean[col], imputed_col, df_mask_in)
            # If the r-squared of imputed data is over 0.7, keep the imputation
            if r2 > error_threshold:
                if verbose:
                    print(f"Using imputed values for {col}: RMSE is {rmse_scaled},  R² is {r2}\n\n")
                    
                r2_list[col] = r2
                imputed_df = pd.concat([imputed_df, imputed_col], axis=1)

            # Otherwise, don't impute that sub
            else:
                if verbose: 
                    print(f"Skipping col {col}, R-squared of imputation was below {error_threshold}\n\n")
                nonimputed_df = pd.concat([nonimputed_df, df_clean[col]], axis=1)
                nonimputed_cols.append(col)
           
    output_df = pd.concat([nonimputed_df, imputed_df], axis=1)
    return output_df, r2_list, imputed_cols, nonimputed_cols

