import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.encoder = None
        self.categorical_dims = {}
        self.column_order = None  # 학습 데이터의 컬럼 순서 저장

    def fit(self, df):
        """범주형 변수의 레이블 인코딩 정보를 학습"""
        # 모든 데이터를 문자열로 변환
        df = df.astype(str)
        
        # OrdinalEncoder 초기화 및 학습
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.encoder.fit(df)
        
        # 각 변수별 고유값 개수를 저장
        for i, column in enumerate(df.columns):
            unique_count = len(np.unique(df[column]))
            self.categorical_dims[column] = unique_count + 1  # +1 to include unknown
        
        # 학습 데이터의 컬럼 순서 저장
        self.column_order = df.columns.tolist()

    def transform(self, df):
        """데이터프레임을 텐서 딕셔너리로 변환"""
        # 누락된 컬럼을 'Missing'으로 채우고, 학습 데이터의 컬럼 순서로 정렬
        for col in self.column_order:
            if col not in df.columns:
                df[col] = 'Missing'
        df = df[self.column_order]

        # 모든 데이터를 문자열로 변환
        df = df.astype(str)
        
        # OrdinalEncoder를 사용하여 변환
        encoded_array = self.encoder.transform(df)

        # unknown 값(-1)을 각 변수의 고유 범주 개수로 변경
        for i, column in enumerate(df.columns):
            encoded_array[:, i] = np.where(
                encoded_array[:, i] == -1, self.categorical_dims[column] - 1, encoded_array[:, i]
            )

        # 변환된 값을 텐서로 반환
        result = {
            col: torch.LongTensor(encoded_array[:, i].astype(int))
            for i, col in enumerate(df.columns)
        }
        return result