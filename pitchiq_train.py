import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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
        self.model: RandomForestClassifier| XGBClassifier | None = None
        self.label_encoder = LabelEncoder()
        self.param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
        }
        self.xgb_param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.3],
        }
        self.xgb_param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "n_estimators": [300, 500]
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

        self.train = self.data[self.data['game_date'].isin(train_games)].copy()
        self.test = self.data[self.data['game_date'].isin(test_games)].copy()

        # encoding pitch_type labels for XGBoost
        self.train['pitch_type_encoded'] = self.label_encoder.fit_transform(self.train['pitch_type'])
        self.test['pitch_type_encoded'] = self.label_encoder.transform(self.test['pitch_type'])

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

    def train_xgboost(self):
        X_train = self.train.drop(columns=['pitch_type', 'pitch_type_encoded', 'game_date'])
        y_train = self.train['pitch_type_encoded']

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        print("Creating XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=20,
            verbose=False,
        )

    def grid_search(self, cv:int =3, scoring: str ='accuracy', verbose:int | bool =1):
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

        grid.fit(X_train, y_train)
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_:.4f}")
        self.model = grid.best_estimator_
        self.save_model(filename="random_forest_pitchiq_tuned.pkl")

    def random_search(self, n_iter:int =20, cv:int =3, scoring: str ='accuracy', verbose:int | bool =1):
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

        rand.fit(X_train, y_train)
        print(f"Best parameters: {rand.best_params_}")
        print(f"Best score: {rand.best_score_:.4f}")
        self.model = rand.best_estimator_
        self.save_model(filename="random_forest_pitchiq_tuned.pkl")

    def random_search_xgboost(self, n_iter:int=25, cv:int=3, scoring: str ='accuracy', verbose:int | bool =1):
        print("Running random search with XGBoost...")
        self.train_test_split()
        X_train = self.train.drop(columns=['pitch_type', 'pitch_type_encoded', 'game_date'])
        y_train = self.train['pitch_type_encoded']

        xgb = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        rand = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=self.xgb_param_dist,
            n_iter=n_iter,
            cv=cv,
            verbose=verbose,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1
        )

        rand.fit(X_train, y_train)

        print(f"Best parameters: {rand.best_params_}")
        print(f"Best score: {rand.best_score_:.4f}")
        self.model = rand.best_estimator_
        self.save_model(filename="xgb_pitchiq_tuned.pkl")

    def grid_search_xgboost(self, cv:int =3, scoring: str ='accuracy', verbose:int | bool =1):
        print("Running grid search with XGBoost...")
        self.train_test_split()
        X_train = self.train.drop(columns=['pitch_type', 'pitch_type_encoded', 'game_date'])
        y_train = self.train['pitch_type_encoded']

        xgb = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=self.xgb_param_grid,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_:.4f}")
        self.model = grid.best_estimator_
        self.save_model(filename="xgb_pitchiq_tuned.pkl")


    def save_model(self, filename: str ='random_forest_pitchiq.pkl') -> None:
        joblib.dump(self.model, f'models/{filename}')
        print(f"Saved model to models/{filename}")

    def evaluate_model(self, show_plot: bool = False) -> None:
        """
        Evaluate the trained model. This handles both:
          - RandomForest trained on string labels
          - XGBoost trained on integer-encoded labels (uses self.label_encoder)

        It prints basic metrics, top-k accuracy, baselines and optionally shows plots.
        """
        # prepare test features and true labels (strings)
        X_test = self.test.drop(columns=['pitch_type', 'pitch_type_encoded', 'game_date'], errors='ignore')
        y_test = self.test['pitch_type'].astype(str)

        print("\n===== MODEL PERFORMANCE =====")
        # compute y_pred (string labels) and y_proba (probabilities)
        y_pred_str, y_proba = self._predict_and_proba(X_test)
        self._print_basic_metrics_from_preds(y_test, y_pred_str)
        self._print_top_k_accuracy_from_proba(y_test, y_proba, k=2)

        print("\n===== BASELINE COMPARISONS =====")
        self._print_baseline_most_frequent(y_test, self.train['pitch_type'])
        self._print_baseline_last_pitch(y_test)
        self._print_baseline_count_only(y_test, self.train['pitch_type'])

        if show_plot:
            print("\n===== CONFUSION MATRIX =====")
            self._plot_confusion_matrix_from_preds(y_test, y_pred_str)

    def _predict_and_proba(self, X):
        """
        Unified predict / predict_proba wrapper that returns:
          - y_pred_str: predicted labels as strings (same domain as original pitch_type)
          - y_proba: numpy array of shape (n_samples, n_classes) with probabilities
        Handles both RF (string labels) and XGB (integer labels + self.label_encoder).
        """
        # Raw predictions from model
        raw_pred = self.model.predict(X)

        # Determine whether raw_pred is integer-encoded
        if np.issubdtype(np.asarray(raw_pred).dtype, np.integer):
            # assume label_encoder was fitted in train_test_split
            try:
                y_pred_str = self.label_encoder.inverse_transform(raw_pred)
            except Exception:
                # fallback: cast to str if decode fails
                y_pred_str = raw_pred.astype(str)
        else:
            # already string labels
            y_pred_str = raw_pred.astype(str)

        # Build probability matrix
        # model.classes_ indicates class ordering for predict_proba columns
        try:
            proba = self.model.predict_proba(X)
        except Exception:
            # If model does not support predict_proba, emulate one-hot from predictions
            n = len(y_pred_str)
            classes = self.model.classes_
            proba = np.zeros((n, len(classes)))
            # find index of each predicted class and set 1.0
            for i, p in enumerate(raw_pred):
                # if integer-coded, map via label_encoder to class index
                if np.issubdtype(np.asarray(p).dtype, np.integer):
                    # model.classes_ are likely integer labels too -> find index
                    idx = np.where(self.model.classes_ == p)[0]
                else:
                    idx = np.where(self.model.classes_ == p)[0]
                if idx.size:
                    proba[i, idx[0]] = 1.0

        # If model.classes_ are integer-coded, convert class ordering to string labels for downstream use
        if np.issubdtype(np.asarray(self.model.classes_).dtype, np.integer):
            try:
                classes_str = self.label_encoder.inverse_transform(self.model.classes_)
            except Exception:
                classes_str = self.model.classes_.astype(str)
        else:
            classes_str = self.model.classes_.astype(str)

        # Reorder proba columns to correspond to classes_str and return both
        # We'll return proba and also attach classes_str as attribute for helper usage
        proba_with_classes = proba  # columns correspond to self.model.classes_
        # But callers may need classes_str -> return both
        return y_pred_str, (proba_with_classes, classes_str)

    def _print_basic_metrics_from_preds(self, y_true, y_pred_str):
        print(f"Accuracy: {accuracy_score(y_true, y_pred_str):.4f}")
        print(classification_report(y_true, y_pred_str, zero_division=0))

    def _print_top_k_accuracy_from_proba(self, y_true, proba_and_classes, k=2):
        """
        proba_and_classes: tuple (proba_array, classes_str_array)
        """
        proba, classes_str = proba_and_classes
        # get top-k column indices and map to string labels
        top_k_idx = np.argsort(proba, axis=1)[:, -k:]
        # map indices -> labels
        top_k_labels = classes_str[top_k_idx]  # shape (n_samples, k)
        # compute top-k accuracy
        topk_correct = [
            y_true.iloc[i] in top_k_labels[i]
            for i in range(len(y_true))
        ]
        topk_acc = np.mean(topk_correct)
        print(f"Top-{k} Accuracy: {topk_acc:.4f}")

    def _plot_confusion_matrix_from_preds(self, y_true, y_pred_str):
        cm = confusion_matrix(y_true, y_pred_str, labels=np.unique(np.concatenate([y_true.unique(), y_pred_str])))
        labels = np.unique(np.concatenate([y_true.unique(), y_pred_str]))
        plt.figure(figsize=(8, 6))
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm.astype(float) / row_sums
        sns.heatmap(
            cm_normalized,
            annot=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            fmt=".2f"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.show()


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

    def run(self, tune=False, tuning_method='grid', model_type='xgb'):
        self.train_test_split()

        if tune:
            if model_type == 'xgb':
                if tuning_method == 'grid':
                    self.grid_search_xgboost()
                elif tuning_method == 'random':
                    self.random_search_xgboost()
                else:
                    raise ValueError("Invalid tuning method for XGBoost.")
            else:
                # RF tuning
                if tuning_method == 'grid':
                    self.grid_search()
                elif tuning_method == 'random':
                    self.random_search()
        else:
            if model_type == 'xgb':
                self.train_xgboost()
            else:
                self.train_model()

        self.evaluate_model(show_plot=True)

if __name__ == "__main__":
    # these hyperparameters give ~57% accuracy
    trainer = ModelTrainer(random_state=42)
    trainer.run(tune=True, tuning_method="random", model_type='RF')