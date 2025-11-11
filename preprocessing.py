from pybaseball import statcast_pitcher, playerid_lookup, cache
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

cache.enable()  # Enable caching to speed up repeated queries

with open("config.json", "r") as config_file:
    config = json.load(config_file)

fullname = config.get("fullname", "Kevin Gausman")
start_dt = config.get("start_dt", "2025-03-27")
end_dt = config.get("end_dt", "2025-11-01")

firstname, lastname = fullname.split(" ", 1)

playerid = playerid_lookup(lastname, firstname, fuzzy=True)['key_mlbam'].iloc[0]

data = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=playerid) 
print(f"Generating .csv files in the pitcher_data directory...")

# begin preprocessing
data.sort_values(['game_date', 'at_bat_number', 'pitch_number'], inplace=True)

data['run_diff'] = data['fld_score'] - data['bat_score']  # create run_diff column

# create lag columns
group_cols = ['game_date', 'at_bat_number']
for col in ['release_speed', 'release_spin_rate'] :
    data[f'prev_{col}'] = data.groupby(group_cols)[col].shift(1)

lag_columns = []
for lag in [1, 2, 3] :
    lag_col_name = f'prev_pitch_type_lag{lag}'
    data[lag_col_name] = (
        data.groupby(group_cols)['pitch_type']
        .shift(lag)
        .fillna('None')   # keep instead of dropping
    )
    lag_columns.append(lag_col_name)
    

data.dropna(subset=lag_columns, inplace=True)
one_hot_columns = lag_columns + ['stand', 'p_throws']
data = pd.get_dummies(data, columns=one_hot_columns) # one-hot encoding for these columns

data['runners_on'] = data[['on_1b', 'on_2b', 'on_3b']].sum(axis=1)

# numerical feature processing
numerical_cols = ['prev_release_speed', 'prev_release_spin_rate', 
    'balls', 'strikes', 'outs_when_up', 'at_bat_number', 'pitch_number',
    'run_diff', 'runners_on']

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

keep_cols = ['pitch_type', 'game_date', 'balls', 'strikes', 'outs_when_up', 'inning', 'p_throws', 'stand',
    'at_bat_number', 'pitch_number',
    'run_diff', 'runners_on',
    'prev_release_speed', 'prev_release_spin_rate'
]
prefixes = ['prev_pitch_type_', 'stand_', 'p_throws_']
extra_cols = [c for c in data.columns if any(c.startswith(p) for p in prefixes)]
data = data[[col for col in keep_cols if col in data.columns] + extra_cols]

data.to_csv(f"pitcher_data/{lastname}_processed.csv", index=False)

print(f"Files generated.")