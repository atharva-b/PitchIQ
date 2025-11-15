import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:

    def __init__(self, processed_csv: str ='pitcher_data/Gausman_processed.csv', df: pd.DataFrame | None = None,
                 test_size: float = 0.2, n_estimators: int = 200, max_depth: int = 10, min_samples_split: int=5,
                 random_state: int = 34, max_features: str = 'sqrt'):
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.model: RandomForestClassifier | None = None
        self.param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
        }
        
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
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        print("Creating model...")
        self.model.fit(X_train, y_train)

    def grid_search(self, cv:int =3, scoring: str ='accuracy', verbose:int | bool =1):
        from sklearn.model_selection import GridSearchCV
        self.train_test_split()
        X_train = self.train.drop(columns=['pitch_type', 'game_date'])
        y_train = self.train['pitch_type']
        grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1
        )
        print("Tuning model using GridSearchCV...")
        print(self.train['pitch_type'].unique())
        print(self.train['game_date'].map(type).unique())
        grid.fit(X_train, y_train)
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_:.4f}")
        self.model = grid.best_estimator_
        self.save_model(filename="random_forest_pitchiq_tuned.pkl")

    def random_search(self, n_iter:int =20, cv:int =3, scoring: str ='accuracy', verbose:int | bool =1):
        from sklearn.model_selection import RandomizedSearchCV
        self.train_test_split()
        X_train = self.train.drop(columns=['pitch_type', 'game_date'])
        y_train = self.train['pitch_type']
        rand = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_distributions=self.param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1,
            random_state=self.random_state
        )
        print("Tuning model using RandomizedSearchCV...")
        print(self.train['pitch_type'].unique())
        print(self.train['game_date'].map(type).unique())
        rand.fit(X_train, y_train)
        print(f"Best parameters: {rand.best_params_}")
        print(f"Best score: {rand.best_score_:.4f}")
        self.model = rand.best_estimator_
        self.save_model(filename="random_forest_pitchiq_tuned.pkl")

    def save_model(self, filename: str ='random_forest_pitchiq.pkl') -> None:
        joblib.dump(self.model, f'models/{filename}')
        print(f"Saved model to models/{filename}")

    def evaluate_model(self, show_plot:bool =False) -> None:
        X_test = self.test.drop(columns=['pitch_type', 'game_date'])
        y_test = self.test['pitch_type']
        y_pred = self.model.predict(X_test)
        print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # Top 3 accuracy
        probs = self.model.predict_proba(X_test)
        top2 = np.argsort(probs, axis=1)[:, -2:]
        top2_accuracy = np.mean([y_test.iloc[i] in self.model.classes_[top2[i]] for i in range(len(y_test))])
        print(f"Top-2 Accuracy: {top2_accuracy:.4f}")

        importances = pd.Series(self.model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        print(importances.head(15))

        # create plot to show confusion matrix
        if show_plot:
            cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Pitch type Confusion Matrix")
            plt.show()

    def run(self, tune:bool =False, tuning_method:str ='grid') -> None:
        self.train_test_split()
        if tune:
            if tuning_method == 'grid':
                self.grid_search()
            elif tuning_method == 'random':
                self.random_search()
            else:
                raise ValueError("Invalid tuning method, use grid or random")
        else:
            self.train_model()
        self.evaluate_model(show_plot=True)

if __name__ == "__main__":
    # these hyperparameters give ~57% accuracy
    trainer = ModelTrainer(n_estimators=400, max_depth=10, min_samples_split=5, random_state=42)
    trainer.run()