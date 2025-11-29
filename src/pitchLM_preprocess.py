import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pitchiq_preprocess import PitcherInfo, PitcherData
import pandas as pd
import os

class PreprocessPitchLM:

    def __init__(
            self,
            pitcher_data: PitcherData,
            vocab: dict[str, int]=None):
        self.pitcher_data: PitcherData = pitcher_data
        self.raw: pd.DataFrame | None = None
        self.sequences: list[list[dict]] | None = None
        self.vocab: dict[str,int] = vocab or {}
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.token2id: dict[str, int] = {}
        self.id2token: dict[str, int] = {}
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.numerical_features = [
            'release_speed',
            'release_spin_rate',
            'run_diff',
            'n_priorpa_thisgame_player_at_bat',
            'pitcher_days_since_prev_game',
        ]
        self.categorical_features = ['pitch_type', 'stand', 'p_throws']

    def load(self):
        self.raw = self.pitcher_data.data.copy()
        print(self.raw['bat_score'].unique())
        print(self.raw['fld_score'].unique())
        print((self.raw['fld_score'] - self.raw['bat_score']).describe())
        self.raw.sort_values(['game_date', 'at_bat_number', 'pitch_number'], inplace=True)
        return self.raw

    def build_pitch_events(self, df):
        cols = [
            'pitch_type',
            'release_speed',
            'release_spin_rate',
            'balls',
            'strikes',
            'outs_when_up',
            'stand',
            'p_throws',
            'at_bat_number',
            'pitch_number',
            'on_1b',
            'on_2b',
            'on_3b',
            'bat_score',
            'fld_score',
        ]
        return df[cols].to_dict(orient='records')

    def group_into_sequences(self):
        grouped = self.raw.groupby(['game_date', 'at_bat_number'])
        sequences = []
        for _, group in grouped:
            seq = group.sort_values("pitch_number").to_dict(orient='records')
            sequences.append(seq)

        self.sequences = sequences
        return sequences

    def build_vocab(self):
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        for feat in self.categorical_features:
            unique_vals = set()
            for seq in self.sequences:
                for ev in seq:
                    if feat in ev and pd.notnull(ev[feat]):
                        unique_vals.add(ev[feat])

            for val in sorted(unique_vals):
                key = f"{feat}_{val}"
                if key not in self.vocab:
                    self.vocab[key] = len(self.vocab)

        self.token2id = self.vocab
        self.id2token = {idx: token for token, idx in self.vocab.items()}
        return self.vocab

    def normalize_num_features(self):
        all_numerical = []
        for seq in self.sequences:
            for ev in seq:
                row = []
                for feature in self.numerical_features:
                    val = ev.get(feature)
                    if pd.notnull(val):
                        row.append(float(val))
                    else:
                        row.append(np.nan)
                all_numerical.append(row)

        if all_numerical:
            data_array = np.array(all_numerical)
            for col in range(data_array.shape[1]):
                col_data = data_array[:, col]
                col_median = np.nanmedian(col_data)
                if np.isnan(col_median):
                    col_median = 0.0
                data_array[np.isnan(data_array[:, col]), col] = col_median

            self.scaler.fit(data_array)

    def encode_event(self, event, debug:bool = False):
        # compute run differential
        bat_score = event.get('bat_score')
        field_score = event.get('fld_score')
        if pd.notnull(bat_score) and pd.notnull(field_score):
            event['run_diff'] = int(field_score) - int(bat_score)
        else:
            event['run_diff'] = 0

        if debug:
            print(f"DEBUG: Raw run_diff = {event['run_diff']}")

        features = {}

        for cat_feature in self.categorical_features:
            val = event.get(cat_feature)
            if pd.notnull(val):
                key = f"{cat_feature}_{val}"
                features[cat_feature] = self.vocab.get(key, self.vocab['<UNK>'])
            else:
                features[cat_feature] = self.vocab['<UNK>']

        num_vals = []
        for num_feature in self.numerical_features:
            val = event.get(num_feature)
            if pd.notnull(val):
                num_vals.append(float(val))
            else:
                num_vals.append(np.nan)

        if num_vals:
            num_vals_clean = [0.0 if np.isnan(v) else v for v in num_vals]
            normalized = self.scaler.transform([num_vals_clean])[0]
            for i, feature in enumerate(self.numerical_features):
                features[feature] = np.clip(normalized[i], 0.0, 1.0)

        features['balls'] = int(event.get('balls', 0))
        features['strikes'] = int(event.get('strikes', 0))
        features['outs_when_up'] = int(event.get('outs_when_up', 0))
        runners = (
            int(bool(event.get('on_1b'))) * 4 +
            int(bool(event.get('on_2b'))) * 2 +
            int(bool(event.get('on_3b')))
        )
        features['runners'] = runners
        return features

    def create_sequence_with_context(self, context_len=5):
        X_sequences = []
        y_labels = []

        for seq in self.sequences:
            encoded_seq = [self.encode_event(event) for event in seq]

            for i in range(1, len(encoded_seq)):
                start = max(0, i - context_len)
                context = encoded_seq[start:i]

                while len(context) < context_len:
                    pad_features = {k: 0 if k not in self.categorical_features else self.vocab['<PAD>']
                                    for k in encoded_seq[0].keys()}
                    context.insert(0, pad_features)

                target_pitch_type = seq[i].get('pitch_type')
                if pd.notnull(target_pitch_type):
                    target_key = f"pitch_type_{target_pitch_type}"
                    target_id = self.vocab.get(target_key, self.vocab['<UNK>'])

                    X_sequences.append(context)
                    y_labels.append(target_id)

        return X_sequences, y_labels

    def create_flat_next_pitch_pairs(self):
        X = []
        y = []
        for seq in self.sequences:
            for i in range(len(seq) - 1):
                current_event = self.encode_event(seq[i])
                next_pitch_type = seq[i + 1].get('pitch_type')

                if pd.notnull(next_pitch_type):
                    target_key = f"pitch_type_{next_pitch_type}"
                    target_id = self.vocab.get(target_key, self.vocab['<UNK>'])

                    X.append(current_event)
                    y.append(target_id)

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self, context_len=5, use_sequences=True):
        self.load()
        self.group_into_sequences()
        self.build_vocab()
        self.normalize_num_features()

        if use_sequences:
            X, y = self.create_sequence_with_context(context_len)
        else:
            X, y = self.create_flat_next_pitch_pairs()

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'vocab': self.vocab,
            'token2id': self.token2id,
            'id2token': self.id2token,
            'scaler': self.scaler,
            'num_classes': len([k for k in self.vocab.keys() if k.startswith('pitch_type_')]),
        }

if __name__ == '__main__':
    # Get the path to config.json relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), 'config.json')
    pitcher_data = PitcherData(config_file=config_path)
    pitcher_data.fetch_data()

    processor = PreprocessPitchLM(pitcher_data)
    data = processor.run()

    print('\n=== DEBUG: First Training Sequence Raw run_diff ===')
    if len(data['X_train']) > 0:
        first_seq_raw = processor.sequences[0]
        for i, event in enumerate(first_seq_raw[:6]):
            processor.encode_event(event, debug=True)

    print('\n=== PREPROCESSING SUMMARY ===')
    print(f"Vocabulary size: {len(data['vocab'])}")
    print(f"Number of pitch types: {data['num_classes']}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")

    print('\n=== SAMPLE DATA ===')
    print(f"First training sequence (context):")
    if len(data['X_train']) > 0:
        print(data['X_train'][0])
    print(f"\nCorresponding label (target pitch): {data['y_train'][0]}")
    print(f"Which is: {data['id2token'][data['y_train'][0]]}")

    print('\n=== PITCH TYPE VOCABULARY ===')
    pitch_types = {k: v for k, v in data['vocab'].items()}
    for pitch, idx in sorted(pitch_types.items(), key=lambda x: x[1]):
        print(f"{pitch}: {idx}")
