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

    def evaluate_model(self, show_plot: bool = False) -> None:
        X_test = self.test.drop(columns=['pitch_type', 'game_date'])
        y_test = self.test['pitch_type']
        y_train = self.train['pitch_type']

        print("\n===== MODEL PERFORMANCE =====")
        self._print_basic_metrics(X_test, y_test)
        self._print_top_k_accuracy(X_test, y_test, k=2)

        print("\n===== BASELINE COMPARISONS =====")
        self._print_baseline_most_frequent(y_test, y_train)
        self._print_baseline_last_pitch(y_test)
        self._print_baseline_count_only(y_test, y_train)

        if show_plot:
            print("\n===== CONFUSION MATRIX =====")
            self._plot_confusion_matrix(X_test, y_test)

    def _print_basic_metrics(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    def _print_top_k_accuracy(self, X_test, y_test, k=2):
        probs = self.model.predict_proba(X_test)
        top_k = np.argsort(probs, axis=1)[:, -k:]
        top_k_acc = np.mean([
            y_test.iloc[i] in self.model.classes_[top_k[i]]
            for i in range(len(y_test))
        ])
        print(f"Top-{k} Accuracy: {top_k_acc:.4f}")

    def _print_baseline_most_frequent(self, y_test, y_train):
        most_frequent = y_train.value_counts().idxmax()
        baseline_acc = (y_test == most_frequent).mean()
        print(f"Most Frequent Pitch Baseline: {baseline_acc:.4f}")

    def _print_baseline_last_pitch(self, y_test):
        lag_cols = [c for c in self.test.columns if c.startswith('prev_pitch_type_lag1_')]

        if not lag_cols:
            print("Last Pitch Baseline: N/A (no lag1 columns found)")
            return

        last_pitch = self.test[lag_cols].idxmax(axis=1)
        last_pitch = last_pitch.str.replace("prev_pitch_type_lag1_", "", regex=False)

        baseline_last = (last_pitch == y_test).mean()
        print(f"Last Pitch Baseline Accuracy: {baseline_last:.4f}")

    def _print_baseline_count_only(self, y_test, y_train):
        from sklearn.linear_model import LogisticRegression

        count_features = ['balls', 'strikes']
        clf = LogisticRegression(max_iter=1000).fit(self.train[count_features], y_train)
        preds = clf.predict(self.test[count_features])
        acc = accuracy_score(y_test, preds)

        print(f"Count-only Baseline Accuracy: {acc:.4f}")

    def _plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_, normalize='true')

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_,
            cmap="Blues",
            fmt=".2f"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.show()

    def show_permutation_importances(self, X_test, y_test):
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=self.random_state
        )

        importances = (
            pd.Series(result.importances_mean, index=X_test.columns)
            .sort_values(ascending=False)
        )

        print(importances)

    def show_partial_display_plot(self, X_test, pitch_type: str='FF'):
        from sklearn.inspection import PartialDependenceDisplay
        PartialDependenceDisplay.from_estimator(
            self.model,
            X_test,
            ['balls', 'strikes', 'runners_on'],
            target=pitch_type
        )
        plt.show()

    def show_shap_analysis(self, X_test):
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test)

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