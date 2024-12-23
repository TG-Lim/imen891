import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib
from models.embedding_model import CategoricalAutoencoder
from models.preprocessor import DataPreprocessor
from models.data import CategoricalDataset


seed = 42

torch.set_float32_matmul_precision('high')
torch.random.manual_seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

df = pd.read_csv('./B2B/train_df.csv', encoding='utf-8')

object_data = df.select_dtypes(include=['object'])
object_data.fillna('Missing', inplace=True)
if 'lead_status' in object_data.columns:
    object_data.drop(columns=['lead_status', 'not_converted_reason'], inplace=True)


# DataLoader를 사용하는 train_model 함수
def train_model_with_dataloader(model, dataloader, num_epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    initial_loss = 0.0

    for batch_data in dataloader:
        batch_data = {k: v.to(next(model.parameters()).device) for k, v in batch_data.items()}
        
        # Forward pass
        encoded, decoded = model(batch_data)
        
        # 각 변수별 손실 계산
        loss = 0
        for col in decoded.keys():
            loss += criterion(decoded[col], batch_data[col])
        initial_loss += loss.item()
    initial_loss /= len(dataloader)

    loss_history = [initial_loss]

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_data in dataloader:
            # GPU로 이동
            batch_data = {k: v.to(next(model.parameters()).device) for k, v in batch_data.items()}
            
            # Forward pass
            encoded, decoded = model(batch_data)
            
            # 각 변수별 손실 계산
            loss = 0
            for col in decoded.keys():
                loss += criterion(decoded[col], batch_data[col])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 에포크당 평균 손실 계산
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        pbar.set_postfix({
            'loss': f"{avg_loss:.6f}"
        })
    torch.save(optimizer.state_dict(), './optimizer_state_dict.pth')
    joblib.dump(loss_history, './loss_history.pkl')
# main 함수 수정
def main_with_dataloader(num_epochs):
    # 데이터 전처리
    preprocessor = DataPreprocessor()
    preprocessor.fit(object_data)
    
    # 텐서 데이터 생성
    tensor_data = preprocessor.transform(object_data)

    embedding_dims = {
        column: min(600, int(1.6 * math.pow(preprocessor.categorical_dims[column], 0.56)))
        for column in preprocessor.categorical_dims.keys()
    }

    print("Category dimensions:", preprocessor.categorical_dims)
    print("Embedding dimensions:", embedding_dims)
    
    # GPU 사용 가능하면 GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = CategoricalAutoencoder(
        categorical_dims=preprocessor.categorical_dims,
        preprocessor=preprocessor,
        embedding_dims=embedding_dims,
        hidden_dim=128
    ).to(device)
    
    # Custom Dataset 생성
    dataset = CategoricalDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    start = time.time()
    # 모델 학습
    train_model_with_dataloader(
        model, 
        dataloader, 
        num_epochs=num_epochs, 
        learning_rate=float(5e-3)
    )
    end = time.time()
    print(f"Training Time: {end - start:.2f} seconds")
    return model, preprocessor

if __name__ == "__main__":
    model, preprocessor = main_with_dataloader(num_epochs=50)
    torch.save(model.state_dict(), './model_state_dict.pth')
    joblib.dump(preprocessor, './preprocessor.pkl')