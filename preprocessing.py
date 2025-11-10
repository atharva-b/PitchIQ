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

# create prev_* columns
group_cols = ['game_date', 'at_bat_number']
for col in ['pitch_type', 'release_speed', 'release_spin_rate'] :
    data[f'prev_{col}'] = data.groupby(group_cols)[col].shift(1)

data.dropna(subset=['prev_pitch_type'], inplace=True)
data = pd.get_dummies(data, columns=['prev_pitch_type', 'stand', 'p_throws']) # one-hot encoding for these columns

data['runners_on'] = data[['on_1b', 'on_2b', 'on_3b']].sum(axis=1)

# numerical feature processing
numerical_cols = ['release_speed', 'release_spin_rate', 'prev_release_speed', 'prev_release_spin_rate', 
    'balls', 'strikes', 'outs_when_up', 'at_bat_number', 'pitch_number', 'pitcher_days_since_prev_game',
    'run_diff', 'runners_on']

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

keep_cols = [
    'pitch_type', 'release_speed', 'release_spin_rate',
    'balls', 'strikes', 'outs_when_up', 'inning', 'p_throws', 'stand',
    'at_bat_number', 'pitch_number',
    'pitcher_days_since_prev_game', 'run_diff', 'runners_on'
    'prev_pitch_type', 'prev_release_speed', 'prev_release_spin_rate'
]
data = data[[col for col in keep_cols if col in data.columns]]

data.to_csv(f"pitcher_data/{lastname}_processed.csv", index=False)

print(f"Files generated.")