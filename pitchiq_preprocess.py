from pybaseball import statcast_pitcher, playerid_lookup, cache
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import os


@dataclass
class PitcherInfo: 
    start_dt: str
    end_dt: str
    firstname: str
    lastname: str
    
class PreprocessData :

    def __init__(self, config_file: str ="config.json", pitcher_info: PitcherInfo | None = None):
        self.pitcher_info = pitcher_info or self.read_config_file(config_file)
        self.player_id = None
        self.data: pd.DataFrame | None = None
        self.lag_columns: list[str] = []
    
    # return a PitcherInfo object with all of the information
    def read_config_file(self, config_file:str ="config.json") -> PitcherInfo:
        with open(config_file, "r") as config_file:
            config = json.load(config_file)
        fullname = config.get("fullname", "Kevin Gausman")
        start_dt = config.get("start_dt", "2025-03-27")
        end_dt = config.get("end_dt", "2025-11-01")

        firstname, lastname = fullname.split(" ", 1)

        return PitcherInfo(start_dt=start_dt, end_dt=end_dt, firstname=firstname, lastname=lastname)
    
    def get_player_id(self) -> int:
        if self.player_id is None:
            cache.enable()
            p = playerid_lookup(self.pitcher_info.lastname, self.pitcher_info.firstname, fuzzy=True)
            self.player_id = int(p['key_mlbam'].iloc[0])
        return self.player_id

    def fetch_data(self) -> pd.DataFrame:
        self.data = statcast_pitcher(start_dt=self.pitcher_info.start_dt, 
                                     end_dt=self.pitcher_info.end_dt,
                                     player_id=self.get_player_id())
        return self.data
    
    def create_lag_columns(self, group_cols:list[str] =['game_date', 'at_bat_number']) -> None:
        df = self.data
        lag_columns = []
        for col in ['release_speed', 'release_spin_rate'] :
            new_col = f'prev_{col}'
            df[new_col] = df.groupby(group_cols)[col].shift(1)
        
        for lag in range(1,4):
            lag_col_name = f'prev_pitch_type_lag{lag}'
            df[lag_col_name] = (
                df.groupby(group_cols)['pitch_type']
                .shift(lag)
                .fillna('None')   # keep instead of dropping
            )
            self.lag_columns.append(lag_col_name)
    
    def preprocess(self) -> pd.DataFrame:
        if self.data is None:
            self.fetch_data()
        data = self.data
        data.sort_values(['game_date', 'at_bat_number', 'pitch_number'], inplace=True)
        data['run_diff'] = data['fld_score'] - data['bat_score']  # create run_diff column

        self.create_lag_columns()

        one_hot_columns = [c for c in self.lag_columns if c.startswith('prev_pitch_type_lag')] + ['stand', 'p_throws']
        data = pd.get_dummies(data, columns=one_hot_columns)

        data['runners_on'] = data[['on_1b', 'on_2b', 'on_3b']].sum(axis=1)

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

        self.data = data
        return self.data

    def save(self, out_dir: str ="pitcher_data") -> str:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.pitcher_info.lastname}_processed.csv")
        self.data.to_csv(out_path, index=False)
        return out_path
    
    def run(self, out_dir: str="pitcher_data") -> str:
        self.fetch_data()
        self.preprocess()
        return self.save(out_dir)


if __name__ == "__main__":
    processor = PreprocessData(config_file="config.json")
    out_path = processor.run()
    print(f"Processed data saved to {out_path}")
