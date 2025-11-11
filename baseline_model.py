import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read processed data
data = pd.read_csv('pitcher_data/Gausman_processed.csv')

# train-test split
game_dates = data['game_date'].unique()
split_point = int(0.8 * len(game_dates))
train_games = game_dates[:split_point]
test_games = game_dates[split_point:]

train = data[data['game_date'].isin(train_games)]
test = data[data['game_date'].isin(test_games)]

X_train = train.drop(columns=['pitch_type', 'game_date'])
y_train = train['pitch_type']
X_test = test.drop(columns=['pitch_type', 'game_date'])
y_test = test['pitch_type']

print("Creating model...")
model = RandomForestClassifier (
    n_estimators=200,
    max_depth=10,
    random_state=34,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Saving model...")
joblib.dump(model, 'models/random_forest_pitchiq.pkl')

# do prediction and prelim accuracy numbers
y_pred = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Top 3 accuracy
probs = model.predict_proba(X_test)
top3 = np.argsort(probs, axis=1)[:, -3:]
top3_accuracy = np.mean([y_test.iloc[i] in model.classes_[top3[i]] for i in range(len(y_test))])
print(f"Top-3 Accuracy: {top3_accuracy:.3f}")

importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importances.head(15))


# create plot to show confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Pitch type Confusion Matrix")

plt.show()