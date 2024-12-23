import pandas as pd
import numpy as np
from hgboost import hgboost
import joblib

seed = 42
file_path = 'hgboost_model.pkl'

if __name__ == '__main__':
    hgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, 
                  random_state=seed, verbose=3, gpu=False)
    df = pd.read_csv('./B2B/train_df.csv', encoding='utf-8')
    numerical = df.select_dtypes(include=['float'])
    embeddings = pd.read_csv('./latent_space_results.csv')

    X = pd.concat([numerical, embeddings], axis=1)
    y = df['lead_status'][df['lead_status'] != 3].values
    X.to_csv('./training_data.csv')
    y = np.where(y == 'Converted', 1, 0)
    np.save('./answer_label.npy', y)

    result = hgb.xgboost(X, y, pos_label=1, eval_metric='f1')
    hgb.save(file_path, overwrite=True)
    joblib.dump(result, './train_result.pkl')