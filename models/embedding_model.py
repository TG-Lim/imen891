import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
import math


class CategoricalAutoencoder(nn.Module):
    def __init__(self, categorical_dims, preprocessor, embedding_dims=None, hidden_dim=64):
        super(CategoricalAutoencoder, self).__init__()
        self.preprocessor = preprocessor
        
        if embedding_dims is None:
            embedding_dims = {col: min(600, int(1.6 * math.pow(self.preprocessor.categorical_dims[col], 0.56))) 
                            for col, num_cats in categorical_dims.items()}
        
        self.embedding_dims = embedding_dims
        self.categorical_dims = categorical_dims
        
        # 각 범주형 변수별 임베딩 레이어 생성
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_cats + 1, embedding_dims[col])  # +1 for unknown in each column
            for col, num_cats in categorical_dims.items()
        })

        
        # 전체 임베딩 차원 계산
        total_embedding_dim = sum(embedding_dims.values())
        
        # 통합 인코더
        self.encoder = nn.Sequential(
            nn.Linear(total_embedding_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # 변수별 디코더 생성 - 수정된 부분
        self.decoders = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(hidden_dim, embedding_dims[col]),
                nn.ELU(),
                nn.Linear(embedding_dims[col], num_cats),
                nn.LogSoftmax(dim=1)
            ) for col, num_cats in categorical_dims.items()
        })
        
    def forward(self, input_dict):
        # 각 변수별 임베딩 추출
        embeddings = []
        for col, embedding_layer in self.embeddings.items():
            embedding = embedding_layer(input_dict[col])
            embeddings.append(embedding)
        
        # 모든 임베딩 연결
        combined_embedding = torch.cat(embeddings, dim=1)
        
        # 인코딩
        encoded = self.encoder(combined_embedding)
        
        # 각 변수별 디코딩
        decoded = {
            col: decoder(encoded) 
            for col, decoder in self.decoders.items()
        }
        
        return encoded, decoded
    
    def get_embeddings(self, input_dict):
        """각 변수별 임베딩을 추출하는 메소드"""
        embeddings = {}
        for col, embedding_layer in self.embeddings.items():
            embeddings[col] = embedding_layer(input_dict[col])
        return embeddings