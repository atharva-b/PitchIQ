from pybaseball import statcast_pitcher, playerid_lookup, cache
import json

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
data.to_csv(f"pitcher_data/{lastname}.csv", index=False)

# begin preprocessing
data.sort_values(['game_date', 'at_bat_number', 'pitch_number'], inplace=True)
data.to_csv(f"pitcher_data/{lastname}_sorted.csv", index=False)

group_cols = ['game_date', 'at_bat_number']
for col in ['pitch_type', 'release_speed', 'release_spin_rate'] :
    data[f'prev_{col}'] = data.groupby(group_cols)[col].shift(1)

data.dropna(subset=['prev_pitch_type'], inplace=True)
data.to_csv(f"pitcher_data/{lastname}_with_lag.csv", index=False)

print(f"Files generated.")