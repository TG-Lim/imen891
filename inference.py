import pandas as pd
import numpy as np
from models.preprocessor import DataPreprocessor
import joblib
import torch
from models.embedding_model import CategoricalAutoencoder
from hgboost import hgboost
from sklearn.metrics import f1_score, accuracy_score

train_data_path = './B2B/train_df.csv'
data_path = './B2B/test_df.csv'
answer_column = 'lead_status'
preprocessor_path = './preprocessor.pkl'
device = "cuda" if torch.cuda.is_available() else "cpu"

def align_dtypes(train_df, test_df):
    """
    train 데이터프레임을 기준으로 test 데이터프레임의 데이터 타입을 맞추는 함수
    """
    for column in train_df.columns:
        if column in test_df.columns:
            # train의 데이터 타입에 따라 test를 변환
            if train_df[column].dtype != test_df[column].dtype:
                if train_df[column].dtype == 'object':
                    test_df[column] = test_df[column].astype(str)
                elif train_df[column].dtype == 'int64':
                    test_df[column] = pd.to_numeric(test_df[column], errors='coerce').astype(int)
                elif train_df[column].dtype == 'float64':
                    test_df[column] = pd.to_numeric(test_df[column], errors='coerce').astype(float)
    return test_df

if __name__ == '__main__':
    df = pd.read_csv(data_path, encoding='utf-8')

    original_df = pd.read_csv(train_data_path, encoding='utf-8')

    df = align_dtypes(original_df, df)

    object_df = df.select_dtypes(include=['object'])
    object_df.fillna('Missing')
    object_df.drop(columns=['not_converted_reason'], inplace=True)
    answer_label = object_df[answer_column]
    y_true = np.where(answer_label == 'Converted', 1, 0)
    object_df.drop(columns=[answer_column], inplace=True)

    preprocessor = joblib.load(preprocessor_path)
    tensor_data = preprocessor.transform(object_df)

    model = CategoricalAutoencoder(categorical_dims=preprocessor.categorical_dims, preprocessor=preprocessor, hidden_dim=128)
    model.load_state_dict(torch.load('./model_state_dict.pth', weights_only=True,map_location=device))
    model.eval()

    encoded, _ = model(tensor_data)

    latent_space = pd.DataFrame(encoded.detach().cpu().numpy())

    numerical = df.select_dtypes(include=['float'])
    X = pd.concat([numerical, latent_space], axis=1)

    X.to_csv('./test_data.csv')
    np.save('./y_true_test.npy', y_true)

    hgb = hgboost()
    result = hgb.load()
    y_pred, _ = hgb.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f'accuracy: {accuracy}')
    print(f'f1 score: {f1}')