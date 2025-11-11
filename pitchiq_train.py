import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:

    def __init__(self, processed_csv: str ='pitcher_data/Gausman_processed.csv', df: pd.DataFrame | None = None,
                 test_size: float = 0.2, n_estimators: int = 200, max_depth: int = 10, random_state: int = 34):
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        
        if df is not None:
            self.data = df
        else:
            self.data = pd.read_csv(processed_csv)
    
    # split data into train and test based on game_date
    def train_test_split(self) -> None:
        game_dates = self.data['game_date'].unique()
        split_point = int((1 - self.test_size) * len(game_dates))
        train_games = game_dates[:split_point]
        test_games = game_dates[split_point:]

        self.train = self.data[self.data['game_date'].isin(train_games)]
        self.test = self.data[self.data['game_date'].isin(test_games)]

    def train_model(self) -> None:
        X_train = self.train.drop(columns=['pitch_type', 'game_date'])
        y_train = self.train['pitch_type']
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        print("Creating model...")
        self.model.fit(X_train, y_train)

    def save_model(self, filename: str ='random_forest_pitchiq.pkl') -> None:
        joblib.dump(self.model, f'models/{filename}')
        print(f"Saved model to models/{filename}")

    def evaluate_model(self) -> None:
        X_test = self.test.drop(columns=['pitch_type', 'game_date'])
        y_test = self.test['pitch_type']
        y_pred = self.model.predict(X_test)
        print(f"accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

        # Top 3 accuracy
        probs = self.model.predict_proba(X_test)
        top3 = np.argsort(probs, axis=1)[:, -3:]
        top3_accuracy = np.mean([y_test.iloc[i] in self.model.classes_[top3[i]] for i in range(len(y_test))])
        print(f"Top-3 Accuracy: {top3_accuracy:.3f}")

        importances = pd.Series(self.model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        print(importances.head(15))

        # create plot to show confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Pitch type Confusion Matrix")
        plt.show()

    def run(self) -> None:
        self.train_test_split()
        self.train_model()
        self.save_model()
        self.evaluate_model()

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()