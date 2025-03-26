## Lists of variable names from data dfs in Brighten Dataset V1 & V2
# List of df names
df_names = ['v1_day', 'v2_day', 'v1_week', 'v2_week']
df_mis = ['v1_day_mis','v2_day_mis','v1_week_mis','v2_week_mis']
df_names_with_mis = df_names + df_mis

# Variables to keep consistent over all dfs and to not alter in preprocessing
id_columns=['participant_id','num_id','dt','week','day']

## Variables from Baseline Surveys
mhs_cols = ['mhs_1','mhs_2','mhs_3','mhs_4','mhs_5']
gad_cols = ['gad7_1','gad7_2','gad7_3','gad7_4','gad7_5','gad7_6','gad7_7','gad7_8','gad7_sum','gad_cat']
phq9_base = ['phq9_1_base','phq9_2_base','phq9_3_base','phq9_4_base','phq9_5_base','phq9_6_base','phq9_7_base','phq9_8_base','phq9_9_base','phq9_sum_base']
mania = ['screen_1','screen_2','screen_3','screen_4']
demographics = ['gender','education','working','income_satisfaction','income_lastyear','marital_status','race','age','heard_about_us','device']
alc_cols = ['alc_1','alc_2','alc_3','alc_sum']

## Variables from Weekly Surveys
phq9_cols = ['phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9','phq9_sum', 'phq9_bin']
sds_cols = ['sds_1','sds_2','sds_3','stress','support']
sleep_cols = ['sleep_1','sleep_2','sleep_3']
gic_cols = ['mood_1']

weekly_cols = phq9_cols + sds_cols + sleep_cols + gic_cols

#### Variables from Daily Surveys
phq2_cols = ['phq2_1','phq2_2','phq2_sum']

daily_cols_v1 = ['aggregate_communication', 'call_count',
       'call_duration', 'interaction_diversity', 'missed_interactions',
       'mobility', 'mobility_radius', 'sms_count', 'sms_length',
       'unreturned_calls']

daily_v2_sensor = ['distance_walking', 'hours_active', 'distance_active',
        'came_to_work','distance_powered_vehicle',
       'hours_high_speed_transportation', 'hours_of_sleep',
       'distance_high_speed_transportation',
       'hours_powered_vehicle', 'hours_stationary', 'hours_stationary_nhw',
       'hours_walking', 'location_variance']

daily_v2_phone = ['callDuration_incoming','callCount_missed',
        'callCount_outgoing','callCount_incoming',
       'callDuration_outgoing', 'textCount', 'textCount_received',
       'textCount_sent', 'textLength_received', 'textLength_sent',
       'uniqueNumbers_calls_incoming', 'uniqueNumbers_calls_missed',
       'uniqueNumbers_calls_outgoing', 'uniqueNumbers_texts',
       'uniqueNumbers_texts_received', 'uniqueNumbers_texts_sent']

daily_v2_weather = ['cloud_cover_mean','dew_point_mean',
        'humidity_mean','temp_mean','dew_point_IQR','humidity_IQR',
        'temp_IQR','cloud_cover_IQR','cloud_cover_std','dew_point_std',
        'humidity_std','temp_std','cloud_cover_median','dew_point_median',
        'humidity_median','temp_median','precip_sum']

daily_v2_common = ['distance_walking', 'hours_active', 'distance_active',
        'distance_powered_vehicle','hours_of_sleep','hours_powered_vehicle',
          'hours_stationary', 'hours_stationary_nhw','hours_walking']

daily_misc_cols = 'hours_accounted_for'
mobility_cols = ['mobility','mobility_radius']

all_daily_cols = phq2_cols + daily_cols_v1 + daily_v2_sensor + daily_v2_phone + daily_v2_weather 




# Aggregated variable lists
numeric_cols = daily_cols_v1 + daily_v2_sensor + daily_v2_phone + phq2_cols + phq9_cols + weekly_cols
passive_cols = daily_cols_v1 + daily_v2_sensor + daily_v2_phone
survey_cols = phq2_cols + weekly_cols
