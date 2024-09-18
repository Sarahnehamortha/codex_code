import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

codex_dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data'
codex_feature_columns = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 
                         'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
                         'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
                         'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured', 
                         'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG', 'referral_source', 'class']

codex_thyroid_data = pd.read_csv(codex_dataset_url, names=codex_feature_columns, na_values='?')
codex_thyroid_data.dropna(inplace=True)

codex_features = codex_thyroid_data.drop('class', axis=1)
codex_target = codex_thyroid_data['class']

codex_feature_encoders = {}
for codex_column in codex_features.columns:
    codex_encoder = LabelEncoder()
    codex_features[codex_column] = codex_encoder.fit_transform(codex_features[codex_column].astype(str))
    codex_feature_encoders[codex_column] = codex_encoder

codex_train_features, codex_test_features, codex_train_target, codex_test_target = train_test_split(
    codex_features, codex_target, test_size=0.2, random_state=42)

codex_naive_bayes_classifier = MultinomialNB()
codex_naive_bayes_classifier.fit(codex_train_features, codex_train_target)

codex_test_predictions = codex_naive_bayes_classifier.predict(codex_test_features)

codex_model_accuracy = accuracy_score(codex_test_target, codex_test_predictions)
print(f'Accuracy: {codex_model_accuracy * 100:.2f}%')

print("Predictions: ", codex_test_predictions)
print("Actual:      ", list(codex_test_target))
