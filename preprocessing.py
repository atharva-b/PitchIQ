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
data.to_csv(f"pitcher_data/{lastname}.csv", index=False)