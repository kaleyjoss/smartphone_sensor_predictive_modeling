## Preprocessing functions

############ LOAD in PACKAGES  #############
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from missforest import MissForest
import outliers
from outliers import smirnov_grubbs as grubbs # annoying to get, 
# ^^ i downloaded the .tar.gz file from https://pypi.org/project/outlier-utils/#files 
# & then copied the outliers folder out to the main site-packages directory so it could be accessible 


# Methods from the sklearn module
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


########### Generic functions ###############
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Save a df to csv at a specific filepath, with optional prefix
def save_df(df, data_dir, filename=None):
    if filename:
        df.to_csv(os.path.join(data_dir, f'{filename}.csv'), index=False)
        return os.path.join(data_dir, f'{filename}.csv')
    else:
        df.to_csv(os.path.join(data_dir, f'{df.name}.csv'), index=False)
        return os.path.join(data_dir, f'{df.name}.csv')


def assign_week_numbers(df):
    # Ensure 'dt' is in datetime format
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

    # Calculate week number for each participant and update back to the original DataFrame
    for participant_id in df['participant_id'].unique():
        # Filter the DataFrame by participant_id
        sub_df = df[df['participant_id'] == participant_id].copy()
        
        # Sort by 'dt'
        sub_df = sub_df.sort_values('dt')
        
        # Calculate the week number relative to the first date for this participant
        sub_df['week'] = ((sub_df['dt'] - sub_df['dt'].min()).dt.days // 7).astype(int)
        
        # Update the original DataFrame with the calculated week numbers
        df.loc[df['participant_id'] == participant_id, 'week'] = sub_df['week']

    return df



#### Functions which create categorical variables / binary variables 

def cat_alc(value):
    if value == 0:
        return 0 #'none'
    if 1 <= value <= 3:
        return 1 #'low'
    elif 4 <= value <= 6:
        return 2 #'med'
    elif 7 <= value:
        return 3 #'high'
    else:
        return 'unknown'  # Handle out-of-range values if necessary
def cat_gad(value):
    if 0 <= value <= 5:
        return 0 #'none'
    if 5 <= value <= 10:
        return 1 #'mild'
    elif 11 <= value <= 15:
        return 2 #'moderate'
    elif 16 <= value:
        return 3 #'severe'
    else:
        return #'unknown'  # Handle out-of-range values if necessary
def cat_phq9(value):
    if 0 <= value <= 4:
        return 0 #'none'
    elif 5 <= value <= 9:
        return 1 #'mild'
    elif 10 <= value <= 14:
        return 2 #'moderate'
    elif 15 <= value <= 19:
        return 3 #'mod severe'
    elif 20 <= value <= 27:
        return 4 #'severe'
    else:
        return 'unknown'  # Handle out-of-range values if necessary

def bin_phq9(value):
    if 0 <= value <= 9:
        return 0 #'not depressed'
    elif 10 <= value:
        return 1 #'depressed'


def bin_phq2(value):
    if 0 <= value <= 2:
        return 0 #'not depressed'
    elif 3 <= value:
        return 1 #'depressed
    


def assign_week_numbers(df):
    '''Purpose: 
    assign week numbers as '1,2,3..' for each participant
    relative to the 'dt' for each participant's row
    
    df = pandas dataframe with datetime column 'dt'
    '''
    # Ensure 'dt' is in datetime format
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

    # Calculate week number for each participant and update back to the original DataFrame
    for participant_id in df['participant_id'].unique():
        # Filter the DataFrame by participant_id
        sub_df = df[df['participant_id'] == participant_id].copy()
        
        # Sort by 'dt'
        sub_df = sub_df.sort_values('dt')
        
        # Calculate the week number relative to the first date for this participant
        sub_df['week'] = ((sub_df['dt'] - sub_df['dt'].min()).dt.days // 7).astype(int)
        
        # Update the original DataFrame with the calculated week numbers
        df.loc[df['participant_id'] == participant_id, 'week'] = sub_df['week']

    return df



########## Functions ##############

########## Combine rows from the same day ###############
pd.set_option('future.no_silent_downcasting', True)
def combine_same_day(df):
    '''Purpose: 
    So that different variables recorded at different 
    times on the SAME day don't show up as NA cells
    but instead are all listed in the same row

    dt = datetime variable from the raw data
    dt_date = just the date as a datetime variable
    '''

    days_subjects = []

    # Step 1: Mask by participant
    for subject in df['participant_id'].unique():
        sub_df = df[df['participant_id'] == subject].copy()

        # Step 2: Set datetime index and create a 'dt_date' column
        if sub_df.index.name != 'dt':
            sub_df.set_index('dt', inplace=True)
        sub_df.index = pd.to_datetime(sub_df.index, errors='coerce')
        sub_df['dt_date'] = sub_df.index.date  # Add a 'dt_date' column for grouping by date

        # Step 3: Group by 'dt_date', fill missing values, and reset index to ungroup
        filled_df = (
            sub_df.groupby('dt_date')
            .apply(lambda x: x.ffill().bfill())
            .reset_index(drop=True)
            .infer_objects(copy=False)  # Fix future dtype change issue
        )


        # Step 4: Extract the unique `week` for each `dt_date`
        week_mapping = filled_df.groupby('dt_date')['week'].first()

        # Step 5: Define aggregation functions for numeric and non-numeric data
        aggregation_functions = {
            col: 'mean' if pd.api.types.is_numeric_dtype(filled_df[col]) else 'first'
            for col in filled_df.columns if col not in ['dt_date', 'week']  # Exclude 'dt_date' and 'week' from aggregation
        }

        # Step 6: Group by 'dt_date' again and apply the aggregation functions
        avg_df = filled_df.groupby('dt_date').agg(aggregation_functions).reset_index()

        # Step 7: Reassign the correct integer `week` value based on `dt_date`
        avg_df['week'] = avg_df['dt_date'].map(week_mapping)

        # Step 8: Add this participant's data to the list
        days_subjects.append(avg_df)

    # Step 9: Concatenate all subjects into a new DataFrame
    days_df = pd.concat(days_subjects)

    # Step 10: Rename the dt_date column to 'dt'
    days_df.rename(columns={'dt_date': 'dt'}, inplace=True)

    # Step 11: Return the DataFrame
    return days_df



########## Combine rows from the same day ###############
def combine_same_week(df):
    '''Purpose: 
    So that different variables recorded at different 
    times on the SAME day don't show up as NA cells
    but instead are all listed in the same row

    dt = datetime variable from the raw data
    dt_date = just the date as a datetime variable
    '''

    days_subjects = []

    # Step 1: Mask by participant
    for subject in df['participant_id'].unique():
        sub_df = df[df['participant_id'] == subject].copy()

        # Step 2: Set datetime index and create a 'dt_date' column
        if sub_df.index.name != 'dt':
            sub_df.set_index('dt', inplace=True)
        sub_df.index = pd.to_datetime(sub_df.index, errors='coerce')
        sub_df['dt_date'] = sub_df.index.date  # Add a 'dt_date' column for grouping by date

        # Step 3: Group by 'dt_date', fill missing values, and reset index to ungroup
        filled_df = (
            sub_df.groupby('dt_date')
            .apply(lambda x: x.ffill().bfill())
            .reset_index(drop=True)
            .infer_objects(copy=False)  # Fix future dtype change issue
        )


        # Step 4: Extract the unique `week` for each `dt_date`
        week_mapping = filled_df.groupby('dt_date')['week'].first()

        # Step 5: Define aggregation functions for numeric and non-numeric data
        aggregation_functions = {
            col: 'mean' if pd.api.types.is_numeric_dtype(filled_df[col]) else 'first'
            for col in filled_df.columns if col not in ['dt_date', 'week']  # Exclude 'dt_date' and 'week' from aggregation
        }

        # Step 6: Group by 'dt_date' again and apply the aggregation functions
        avg_df = filled_df.groupby('dt_date').agg(aggregation_functions).reset_index()

        # Step 7: Reassign the correct integer `week` value based on `dt_date`
        avg_df['week'] = avg_df['dt_date'].map(week_mapping)

        # Step 8: Add this participant's data to the list
        days_subjects.append(avg_df)

    # Step 9: Concatenate all subjects into a new DataFrame
    days_df = pd.concat(days_subjects)

    # Step 10: Rename the dt_date column to 'dt'
    days_df.rename(columns={'dt_date': 'dt'}, inplace=True)

    # Step 11: Return the DataFrame
    return days_df

############ CREATE #4 df_alldays & SAVE .CSV ###############
#  days_df -> df_alldays, reindexing each range of dates for a participant to include all dates in that range

def reindex_to_all_days(days_df):
    days_df['dt'] = pd.to_datetime(days_df['dt'])  # Ensure 'dt' is a datetime column

    # Container to store each participant's reindexed data
    reindexed_data = []

    # Iterate over each participant's data
    for participant, group in days_df.groupby('participant_id'):
        # Sort by 'dt' to ensure chronological order
        group = group.sort_values('dt')

        # Generate a full date range from the first to the last date in this participant's data
        full_date_range = pd.date_range(start=group['dt'].min(), end=group['dt'].max(), freq='D')

        # Reindex the group to include all dates in the range
        # This will introduce NaN values for any dates not present in the original data
        group = group.set_index('dt').reindex(full_date_range).reset_index()
        group['participant_id'] = participant  # Add participant ID back to the DataFrame
        group.rename(columns={'index': 'dt'}, inplace=True)  # Rename index back to 'dt'
        


        # Append this participant's reindexed data to the list
        reindexed_data.append(group)

    # Concatenate all reindexed data into a single DataFrame
    days_df_alldays = pd.concat(reindexed_data, ignore_index=True)
    days_df_alldays = assign_week_numbers(days_df_alldays)

    # Now `days_df_alldays` has rows for all dates between each participant's first and last date,
    # with NaNs where data was originally missing.

    # Initialize an empty column for day numbers
    days_df_alldays['day'] = pd.NA

    # Loop over each participant group
    for participant, group in days_df_alldays.groupby('participant_id'):
        # Sort by 'dt' to ensure chronological order
        group = group.sort_values('dt')
        
        # Assign day numbers in sequence
        for count, i in enumerate(group.index):
            # Update the 'day' column in the original DataFrame
            days_df_alldays.loc[i, 'day'] = count

    return days_df_alldays


################## Linear Interpolation function #########################
def add_linear_interpolated_col(input_df, cols_to_interpolate, threshold_percentage, overwrite=False, verbose=False):

    int_label = f'_int{threshold_percentage}'
    input_df['dt'] = pd.to_datetime(input_df['dt'], errors='coerce')
    
    if not any(int_label in col for col in input_df.columns) or overwrite:
        output_df = input_df.copy()
        
        output_df['dt'] = pd.to_datetime(output_df['dt'], errors='coerce')
        
        for var in cols_to_interpolate:
            if verbose==True:
                print(f'Processing variable: {var}')
            int_df = pd.DataFrame()
            var_weeks = 0
            var_weeks_interpolated = 0

            for participant, sub_weeks in output_df.groupby('participant_id'):
                sub_weeks = sub_weeks.sort_values('dt')
                num_unique_weeks = sub_weeks['week'].nunique()  # Count unique weeks
                var_weeks += num_unique_weeks
                
                for week_num, sub_week in sub_weeks.groupby('week'):
                    percent_non_na = (1 - sub_week[var].isna().mean()) * 100

                    if percent_non_na >= threshold_percentage and percent_non_na < 100:
                        if verbose==True:
                            print(f'Participant {participant} | Week {week_num} | Percent non-NA: {round(percent_non_na,3)}\nInterpolating...')
                        sub_week = sub_week.set_index('dt')
                        sub_week[f'{var}{int_label}'] = sub_week[var].interpolate(method='time')
                        percent_non_na_after = (1 - sub_week[f'{var}{int_label}'].isna().mean()) * 100
                        sub_week = sub_week.reset_index()
                        if percent_non_na_after > percent_non_na:
                            if verbose == True:
                                print('Subject interpolated: ', participant)
                                print(f'Before interpolating')
                                print(sub_week[var])
                                print(f'After interpolating')
                                print(sub_week[f'{var}{int_label}'])
                            var_weeks_interpolated += 1
                    else:
                        continue

                    int_df = pd.concat([int_df, sub_week], axis=0)

            print(f'Var: {var} | Total weeks: {var_weeks} | Weeks interpolated: {var_weeks_interpolated}')
            if var_weeks_interpolated > 0: 
                output_df = output_df.merge(
                    int_df[[f'{var}{int_label}', 'participant_id', 'dt']],
                    on=['participant_id', 'dt'], 
                    how='left'
                )

        # # now you need to fill any data from og column into int column since any data which wasn't interpolated
        # # For each row, this method will take the value from original_data ({var})
        # # if it's not NaN. If original_data has NaN, it takes the value from {var}{int_label}.
        for var in cols_to_interpolate:
            var_int_col = var + int_label
            if var_int_col in output_df.columns:
                output_df[f'{var}_int'] = output_df[f'{var}{int_label}'].combine_first(output_df[var])
                output_df = output_df.drop(f'{var}{int_label}', axis=1)    

        return output_df
    
    else:
        print(f'Interpolation at {threshold_percentage}% threshold already exists in input_df.')    



################## Impute using MissForest #########################


def simulate_missing_data(df, missing_percentage=0.1):
    """Randomly remove values from the dataframe."""
    df_missing = df.copy()
    mask = np.random.rand(*df_missing.shape) < missing_percentage
    
    # Convert mask to a Pandas DataFrame
    mask_df = pd.DataFrame(mask, index=df.index, columns=df.columns)
    
    df_missing[mask] = np.nan
    return df_missing, mask_df


# Updated compute_imputation_error provided above
def compute_imputation_error(original_df, imputed_df, mask):
    """
    Compute the mean squared error between original and imputed values,
    but only for those entries where missingness was simulated (according to mask).
    """
    # Make sure the dfs have the same columns/size
    different_columns = len(set(original_df.columns) - set(imputed_df))
    if different_columns > 0:
        print("ERROR: Columns not same:", different_columns)
        return 999
    elif (original_df.shape[0] != imputed_df.shape[0]) or (original_df.shape[1] != imputed_df.shape[1]):
        print(f"ERROR: Not same size. original_df {original_df.shape}, imputed_df {imputed_df.shape}")
        return 999
    else:
        # Convert the mask (a NumPy array) to a DataFrame matching the original DataFrame
        mask_df = pd.DataFrame(mask, index=original_df.index, columns=original_df.columns)
        valid_mask = mask_df & original_df.notna() & imputed_df.notna()
        original_values = original_df[valid_mask].to_numpy().flatten()
        imputed_values = imputed_df[valid_mask].to_numpy().flatten()
        valid_idx = ~np.isnan(original_values) & ~np.isnan(imputed_values)
        original_values = original_values[valid_idx]
        imputed_values = imputed_values[valid_idx]
        # print(f"Valid values for MSE calculation: {len(original_values)}")
        # print(f"Original values: {original_values}")
        # print(f"Imputed values: {imputed_values}")
        if original_values.size == 0:
            return np.nan
        # Evaluate MSE & R-squared
        rmse_value = rmse(original_values, imputed_values)
        rmse_scaled = rmse_value / original_values.mean()
        r2 = sklearn.metrics.r2_score(original_values,imputed_values)
        
        return rmse_scaled, r2

def missforest_imputation(df, cols_to_impute, imputation_threshold=0.6, error_threshold=0.8, verbose=False):
    """
    Impute missing values using MissForest for each subject (identified by 'num_id')
    if the subject has at least imputation_threshold proportion of non-missing data.
    """
    r2_list = {}
    imputed_sub_dfs = []
    imputed_subs = []
    nonimputed_subs = []

    for sub in df['num_id'].unique():
        sub_df = df[df['num_id'] == sub].copy()
        if verbose:
            print(f'For sub {sub}, {sub_df.shape[0]} rows')
        
        # Only impute for subjects with 25 days or more
        if sub_df.shape[0] < 25:
            if verbose:
                print(f"Skipping subject {sub}: all imputation columns are missing.")
            imputed_sub_dfs.append(sub_df)
            nonimputed_subs.append(sub)
            continue
        
        else:
            nonimpute_cols = [col for col in df.columns if col not in cols_to_impute]
            impute_df = sub_df[cols_to_impute].copy()
            
            # Drop columns that are entirely NaN
            fully_missing_cols = impute_df.columns[impute_df.isna().all()]
            if len(fully_missing_cols) > 0:
                impute_df_clean = impute_df.drop(columns=fully_missing_cols)
                nonimpute_cols = nonimpute_cols + list(fully_missing_cols)
                # Skip this subject if all columns were removed
                if impute_df_clean.shape[1] == 0:
                    if verbose:
                        print(f"Skipping subject {sub}: all imputation columns are missing.")
                    imputed_sub_dfs.append(sub_df)
                    nonimputed_subs.append(sub)
                    continue
            else:
                impute_df_clean = impute_df

            # Calculate the proportion of non-missing data in impute_df_clean
            non_missing_proportion = impute_df_clean.notna().mean().mean()
            
            # Check that it's over the imputation threshold (ie 60%, 70% etc)
            if non_missing_proportion >= imputation_threshold:
                df_sim_missing, sim_missing_mask = simulate_missing_data(impute_df_clean, missing_percentage=0.1)
                
                # Drop any columns in df_sim_missing that became fully NaN
                fully_missing_cols = df_sim_missing.columns[df_sim_missing.isna().all()]
                if len(fully_missing_cols) > 0:
                    impute_df_clean = df_sim_missing.drop(columns=fully_missing_cols)
                    sim_missing_mask = pd.DataFrame(sim_missing_mask, index=df_sim_missing.index, 
                                                    columns=df_sim_missing.columns).drop(columns=fully_missing_cols).to_numpy()
                    nonimpute_cols = nonimpute_cols + list(fully_missing_cols)
                
                # Check if all the imputation columns are empty and skip if so
                imputer = MissForest()
                if len(df_sim_missing.columns[df_sim_missing.isna().all()]) > 0:
                    if verbose:
                        print(f"Skipping subject {sub}: all imputation columns are missing.")
                    imputed_sub_dfs.append(sub_df)
                    nonimputed_subs.append(sub)
                    continue
                else:
                    # Otherwise, impute the data
                    df_imputed = imputer.fit_transform(df_sim_missing)
                    df_imputed = pd.DataFrame(df_imputed, columns=impute_df_clean.columns)

                    # Compute the error of imputed data 
                    rmse_scaled, r2 = compute_imputation_error(impute_df_clean, df_imputed, sim_missing_mask)
                    if verbose:
                        #print(f"RMSE (as percentage of value) for subject {sub} is {rmse_scaled}")
                        print(f"R-squared for subject {sub} is {r2}")
                    # If the r-squared of imputed data is over 0.75, keep the imputation
                    if r2 > error_threshold:
                        sub_df_nonimputed = sub_df[nonimpute_cols]
                        sub_df_imputed = sub_df_nonimputed.merge(df_imputed, left_index=True, right_index=True)
                        imputed_sub_dfs.append(sub_df_imputed)
                        imputed_subs.append(sub)
                        r2_list[sub] = r2
                    # Otherwise, don't impute that sub
                    else:
                        if verbose: 
                            print(f"Skipping sub {sub}, R-squred of imputation was below 0.75")
                        imputed_sub_dfs.append(sub_df)
                        nonimputed_subs.append(sub)
                    

                    
            else:
                if verbose:
                    print(f"Skipping subject {sub}: Not enough non-missing data ({non_missing_proportion:.2f}).")
                imputed_sub_dfs.append(sub_df)
                nonimputed_subs.append(sub)

    imputed_df = pd.concat(imputed_sub_dfs)
    return imputed_df, r2_list, imputed_subs, nonimputed_subs



####################### Regress out covariates ##########################
def regress_covariates(df, to_regress_out, to_ignore=None):
    # 2. Define covariates
    to_modify = list(set(df.columns.to_list()) - set(to_regress_out + to_ignore))

    # 3. Extract relevant columns
    X = df[to_regress_out] # covariates
    y = df[to_modify] # numerical columns to do regression on

    # 4. Fit the model
    reg = LinearRegression().fit(X, y)

    # 5. Get predicted values (what ['gender','education','race','age']   would have contributed)
    y_pred = reg.predict(X)

    # 6. Get residuals
    df_residuals = df.copy()
    df_residuals[to_modify] = y - y_pred

    return df_residuals



# Normalize data and Plot histogram of values for each variable & after normalization
# figure out how to drop 0 / 0-inflated? 

num_to_plot = 3

def plot_normalization(df, cols_to_scale, num_to_plot, subject_to_plot=None):
    ''' Plotting the normalization and outlier-removal of select numeric columns to visually inspect non-warping
    df = DataFrame with numeric_cols in columns 
    numeric_cols = list of all columns in df which are numeric
    num_to_plot = int of number of variables to plot

    subject_to_plot: plot also a single subject to see their distribution has not warped. Find 
    find a subject with non-zero values for one or more of the numeric_cols being plotted 
    '''
    scaler = StandardScaler()
    scaled_df = df.copy()
    
    for x_col in cols_to_scale[0:num_to_plot]:
        col_df = scaled_df[[x_col]].dropna()  # Drop NaNs
        ## Find outliers using Smirnov-Grubbs test
        non_outliers_mask = grubbs.test(col_df[[x_col]].to_numpy(), alpha=0.05).flatten()
        
        # Keep only non-outliers
        col_df.loc[:, f'{x_col}_nonoutliers'] = np.where(
            col_df[x_col].isin(non_outliers_mask),
            col_df[x_col],
            np.nan  # Replace outliers with NaN
        )

        # Apply scaling
        scaled_values = scaler.fit_transform(col_df[[f'{x_col}_nonoutliers']])
        col_df.loc[:, f'{x_col}_nonoutliers_scaled'] = scaled_values

        
        # Merge back to the scaled_df
        scaled_df.loc[col_df.index, f'{x_col}_nonoutliers'] = col_df[f'{x_col}_nonoutliers']
        scaled_df.loc[col_df.index, f'{x_col}_nonoutliers_scaled'] = col_df[f'{x_col}_nonoutliers_scaled']
    
        # Plot raw distribution of x_col
        print('This is the raw data')
        sns.histplot(col_df[x_col], kde=True)
        plt.title(f"Distribution of {x_col}")
        plt.show()

        # Check distribution after removing outliers
        print('This is the lesioned data- it should have less outliers')
        sns.histplot(col_df[f'{x_col}_nonoutliers'], kde=True)
        plt.title(f"{x_col} Distribution no outliers ")
        plt.show()

        # Check distribution after scaling & removing outliers
        print('Scaled data for same subject in  original dataframe-- should look exaclty like above with different scale')
        sns.histplot(col_df[f'{x_col}_nonoutliers_scaled'], kde=True)
        plt.title(f"Standard-Scaled {x_col} Distribution no outliers")
        plt.show()

        # Check distribution assigning the non-outliers to days_df
        print('More lesioned data but in new  dataframe-- should look exactly like above')
        sns.histplot(scaled_df[f'{x_col}_nonoutliers_scaled'], kde=True)
        plt.title(f"Days_df Scaled {x_col} Distribution no outliers")
        plt.show()

        ## OPTIONAL: Display plot for a single subject to check the indexing is right
        if subject_to_plot: 
            # Find a subject that has non-zero values for the relevant variables & filter for them
            sub_days_df = scaled_df[scaled_df['participant_id']== subject_to_plot]
            sub_col_df = col_df[col_df['participant_id']==subject_to_plot]

            print('Scaled Data for one subject in modified dataframe')
            plt.plot(sub_col_df['dt'], sub_col_df[f'{x_col}_nonoutliers'], label='days_df', color='blue', marker='o')  
            plt.show()

            print('Lesioned data for same subject in  original dataframe-- should have less outliers')
            plt.plot(sub_days_df['dt'], sub_days_df[x_col], label='sub_days_df', color='red', marker='o')  
            plt.show()
            
            print('Scaled data for same subject in  original dataframe-- should look exaclty like above with different scale')
            plt.plot(sub_days_df['dt'], sub_days_df[f'{x_col}_nonoutliers_scaled'], label='sub_days_df', color='red', marker='o')  
            plt.show()


# Scale data
def normalize_df(df, cols_to_scale):
    scaler = StandardScaler()  # Use a single scaler for all numeric cols
    
    scaled_df = df.copy()
    
    for x_col in cols_to_scale: 
        col_df = scaled_df[[x_col]].dropna()  # Drop NaNs
        ## Find outliers using Smirnov-Grubbs test
        non_outliers_mask = grubbs.test(col_df[[x_col]].to_numpy(), alpha=0.05).flatten()
        
        # Keep only non-outliers
        col_df.loc[:, f'{x_col}_nonoutliers'] = np.where(
            col_df[x_col].isin(non_outliers_mask),
            col_df[x_col],
            np.nan  # Replace outliers with NaN
        )

        # Apply scaling
        scaled_values = scaler.fit_transform(col_df[[f'{x_col}_nonoutliers']])
        col_df.loc[:, f'{x_col}_nonoutliers_scaled'] = scaled_values

        

        # Merge back to the scaled_df
        scaled_df.loc[col_df.index, f'{x_col}_nonoutliers_scaled'] = col_df[f'{x_col}_nonoutliers_scaled']
    
    # drop all raw cols from scaled_df (cols which now have a scaled version with "_nonoutliers_scaled")
    scaled_df = scaled_df.drop(columns=cols_to_scale, errors='ignore')
    # Rename "col_nonoutliers_scaled" to just "col"
    scaled_df.rename(columns=lambda x: x.replace("_nonoutliers_scaled", ""), inplace=True)

    return scaled_df





################# Create lagged variables for predictor  ##################

def create_lag_variables(df, lag_variables, rows_lagged=-1):
    """
    Create lagged versions of selected variables, shifting values forward in time.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    lag_variables (list): List of column names to create lagged versions for.
    days_lagged (int): Number of days to shift the variable forward. -1 is predicting lag_variable of the next
    row (this week/day predicting next week/day), +1 would be predicting lag variable of the previous row (this
    week/day predcting last week/day)

    Returns:
    pd.DataFrame: Original dataframe with added lagged columns.
    """

    # Ensure dt is a datetime object
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values(by=['participant_id', 'dt'])

    # Create a copy to store new lagged values
    lagged_dfs = []

    for col in lag_variables: 
        print(f'\n Adding lag of {rows_lagged} to column: {col} -> {col}_lag{rows_lagged}')
        
        # Filter for non-NA values in the selected column
        col_df = df[['participant_id', 'dt', col]].dropna().copy()
        print(f'col df shape is: {col_df.shape}')

        # Group by participant and shift the variable
        col_df[f'{col}_lag{rows_lagged}'] = col_df.groupby('participant_id')[col].shift(periods=rows_lagged)
        
        lagged_dfs.append(col_df[['participant_id', 'dt',col, f'{col}_lag{rows_lagged}']])

    # Merge all lagged columns back into the original dataframe **once**
    for lagged_df in lagged_dfs:
        df = df.merge(lagged_df, on=['participant_id', 'dt'], how='left')

    return df



def compute_slope_of_feature(group, x_col="day", y_col="feature"):
    """Returns the slope of y_col vs. x_col for the given group (DataFrame)."""
    if len(group) < 2:
        return np.nan  # Can't compute slope with <2 points
    
    x = group[x_col].values
    y = group[y_col].values
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope



def make_wide_df(df, ignore_columns):

    # Start a "final" table with just ignore_columns (unique), so we can merge results in.
    wide_df = df[ignore_columns].drop_duplicates().copy()

    # Create subsets for each time range
    df_week1 = df[df["week"] <= 1]
    df_weeku2 = df[df["week"] <= 2]
    df_weeku4 = df[df["week"] <= 4]
    df_week4 = df[df["week"] == 4]
    # df_week6 = df[df["week"] >= 6]

    # Aggregate desired columns
    columns = list(set(df.columns.to_list()) - set(ignore_columns))

    for feature in columns:
        #print('\nFOR FEATURE: ', feature)
        # 1) Compute stats for Week 1
        # ---------------------------
        # overall average
        w1_avg = (
            df_week1.groupby("num_id")[feature]
            .mean()
            .reset_index(name=f"{feature}_avg_w1")
        )

        
        # 2) Compute stats for Weeks ≤ 2
        # -------------------------------
        wu2_avg = (
            df_weeku2.groupby("num_id")[feature]
            .mean()
            .reset_index(name=f"{feature}_avg_wu2")
        )


        # slope over that time period
        wu2_slope = (
            df_weeku2.groupby("num_id")
            .apply(lambda g: compute_slope_of_feature(g, x_col="day", y_col=feature))
            .reset_index(name=f"{feature}_slope_wu2")
        )
        
        # 3) Compute stats for Weeks ≤ 4
        # -------------------------------
        wu4_avg = (
            df_weeku4.groupby("num_id")[feature]
            .mean()
            .reset_index(name=f"{feature}_avg_wu4")
        )
        
        # slope over that time period
        wu4_slope = (
            df_weeku4.groupby("num_id")
            .apply(lambda g: compute_slope_of_feature(g, x_col="day", y_col=feature))
            .reset_index(name=f"{feature}_slope_wu4")
        )

        # 4) Compute stats for Week = 4
        w4_avg = (
            df_week4.groupby("num_id")[feature]
            .mean()
            .reset_index(name=f"{feature}_avg_w4")
        )

        # 6)

        # 4) Combine all week-range stats for this feature
        # ------------------------------------------------
        feature_stats = (
            w1_avg
            .merge(wu2_avg, on="num_id", how="outer")
            .merge(wu2_slope, on="num_id", how="outer")
            .merge(wu4_avg, on="num_id", how="outer")
            .merge(wu4_slope, on="num_id", how="outer")
            .merge(w4_avg, on="num_id", how="outer")
        )        
        
        # 5) Merge into our final DataFrame
        # ---------------------------------
        wide_df = wide_df.merge(feature_stats, on="num_id", how="outer")

    if 'dt' in wide_df.columns:
        wide_df = wide_df.drop(columns=['dt'], axis=1)
    if 'week' in wide_df.columns:
        wide_df = wide_df.drop(columns=['week'], axis=1)
    if 'day' in wide_df.columns:
            wide_df = wide_df.drop(columns=['day'], axis=1)
    wide_df = wide_df.drop_duplicates()

        
    # "wide_df" now has columns for every feature and all the created averages
    #print(wide_df.head())

    return wide_df


def round_vars_phq9(wide_df):
    # make sure the cat is still classifier-friendly (discrete vars)    
    wide_df['phq9_bin_avg_w1'] = wide_df['phq9_bin_avg_w1'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq9_bin_avg_wu2'] = wide_df['phq9_bin_avg_wu2'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq9_bin_avg_wu4'] = wide_df['phq9_bin_avg_wu4'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq9_bin_avg_w4'] = wide_df['phq9_bin_avg_w4'].apply(lambda x: int(x) if pd.notna(x) else x)

    return wide_df

def round_vars_phq2(wide_df):
    # make sure the cat is still classifier-friendly (discrete vars)
    wide_df['phq2_bin_avg_w1'] = wide_df['phq2_bin_avg_w1'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq2_bin_avg_wu2'] = wide_df['phq2_bin_avg_wu2'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq2_bin_avg_wu4'] = wide_df['phq2_bin_avg_wu4'].apply(lambda x: int(x) if pd.notna(x) else x)
    wide_df['phq2_bin_avg_w4'] = wide_df['phq2_bin_avg_w4'].apply(lambda x: int(x) if pd.notna(x) else x)

    return wide_df


def person_centered_df(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Centers each feature by subtracting the person's mean for that feature.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with columns including 'person', and specified features over time (rows).
        features (list): List of feature column names to be centered.
        
    Returns:
        pd.DataFrame: DataFrame with person-centered features.
    """
    # Calculate person-wise mean for each feature
    person_means = df.groupby('person')[features].transform('mean')
    
    # Subtract the person's mean from each feature
    df_centered = df.copy()
    df_centered[features] = df[features] - person_means
    
    return df_centered

