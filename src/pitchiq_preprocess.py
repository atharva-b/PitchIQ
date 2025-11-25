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


class PitcherData:
    def __init__(self, config_file: str = "config.json", pitcher_info: PitcherInfo | None = None):
        self.pitcher_info = pitcher_info or self.read_config_file(config_file)
        self.player_id: int | None = None
        self.data: pd.DataFrame | None = None

    def read_config_file(self, config_file: str = "config.json") -> PitcherInfo:
        with open(config_file, "r") as f:
            config = json.load(f)

        fullname = config.get("fullname", "Kevin Gausman")
        start_dt = config.get("start_dt", "2025-03-27")
        end_dt = config.get("end_dt", "2025-11-01")

        firstname, lastname = fullname.split(" ", 1)
        return PitcherInfo(start_dt=start_dt, end_dt=end_dt, firstname=firstname, lastname=lastname)

    def get_player_id(self) -> int:
        if self.player_id is None:
            cache.enable()
            lookup = playerid_lookup(self.pitcher_info.lastname, self.pitcher_info.firstname, fuzzy=True)
            self.player_id = int(lookup["key_mlbam"].iloc[0])
        return self.player_id

    def fetch_data(self) -> pd.DataFrame:
        self.data = statcast_pitcher(
            start_dt=self.pitcher_info.start_dt,
            end_dt=self.pitcher_info.end_dt,
            player_id=self.get_player_id(),
        )
        return self.data


class PreprocessClassifier:
    def __init__(self, pitcher_data: PitcherData):
        self.pitcher_data = pitcher_data
        self.data: pd.DataFrame | None = None
        self.lag_columns: list[str] = []

    # --- FEATURE ENGINEERING ---

    def create_lag_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["game_date", "at_bat_number"]

        # Numerical lags
        for col in ["release_speed", "release_spin_rate"]:
            data[f"prev_{col}"] = data.groupby(group_cols)[col].shift(1)

        # Pitch-type sequence lags
        for lag in range(1, 4):
            name = f"prev_pitch_type_lag{lag}"
            data[name] = (
                data.groupby(group_cols)["pitch_type"]
                .shift(lag)
                .fillna("NoPrevPitch")
            )
            self.lag_columns.append(name)

        return data

    def add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["run_diff"] = data["fld_score"] - data["bat_score"]
        data["runners_on"] = data[["on_1b", "on_2b", "on_3b"]].sum(axis=1)
        return data

    # --- ENCODING ---

    def encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = self.lag_columns + ["stand", "p_throws"]
        return pd.get_dummies(data, columns=categorical_cols)

    # --- SCALING ---

    def scale_numericals(self, data: pd.DataFrame) -> pd.DataFrame:
        numerical_cols = [
            "prev_release_speed",
            "prev_release_spin_rate",
            "balls",
            "strikes",
            "outs_when_up",
            "at_bat_number",
            "pitch_number",
            "run_diff",
            "runners_on",
            "inning",
        ]

        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data

    # --- FINAL COLUMN SELECTION ---

    def select_final_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        base_cols = [
            "pitch_type",
            "game_date",
            "balls",
            "strikes",
            "outs_when_up",
            "inning",
            "p_throws",
            "stand",
            "at_bat_number",
            "pitch_number",
            "run_diff",
            "runners_on",
            "prev_release_speed",
            "prev_release_spin_rate",
        ]

        prefix_cols = [
            c for c in data.columns
            if c.startswith("prev_pitch_type_")
            or c.startswith("stand_")
            or c.startswith("p_throws_")
        ]

        final_cols = [c for c in base_cols if c in data.columns] + prefix_cols
        return data[final_cols]

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def preprocess(self) -> pd.DataFrame:
        raw = self.pitcher_data.data.copy()
        raw.sort_values(["game_date", "at_bat_number", "pitch_number"], inplace=True)

        raw = self.create_lag_columns(raw)
        raw = self.add_derived_features(raw)
        raw = raw.dropna(subset=["pitch_type"])

        raw = self.encode_categoricals(raw)
        raw = self.scale_numericals(raw)
        raw = self.select_final_columns(raw)

        self.data = raw
        return raw

    def save(self, out_dir: str = "pitcher_data") -> str:
        os.makedirs(out_dir, exist_ok=True)
        name = f"{self.pitcher_data.pitcher_info.lastname}_processed.csv"
        path = os.path.join(out_dir, name)
        self.data.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    pitcher = PitcherData(config_file="../config.json")
    pitcher.fetch_data()

    processor = PreprocessClassifier(pitcher)
    processor.preprocess()
    path = processor.save(out_dir="../pitcher_data")
    print(f"Processed data saved to {path}")
