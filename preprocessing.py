from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher_pitch_arsenal
from pybaseball import cache

cache.enable()  # Enable caching to speed up repeated queries

fullname = input("Enter player's full name (First Last): ")
if(fullname == ""):
    fullname = "Jacob deGrom"
firstname, lastname = fullname.split(" ", 1)



start_dt = input("Enter start date (YYYY-MM-DD): ")
end_dt = input("Enter end date (YYYY-MM-DD): ")

if(start_dt == ""):
    start_dt = "2025-01-01"
if (end_dt == ""):
    end_dt = "2025-09-30"

playerid = playerid_lookup(lastname, firstname, fuzzy=True)['key_mlbam'].iloc[0]

data = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=playerid) 
data.to_csv("pitcher_data/gausman.csv", index=False)

# print(statcast_pitcher_pitch_arsenal(2025, arsenal_type="n_"))

# #Or print them one per line
# print("\nColumn names (one per line):")
# for col in data.columns:
#     print(col)


