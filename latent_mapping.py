from models.embedding_model import CategoricalAutoencoder
from models.preprocessor import DataPreprocessor
from models.data import CategoricalDataset
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader

seed = 42

torch.set_float32_matmul_precision('high')
torch.random.manual_seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model_path = './model_state_dict.pth'
preprocessor_path = './preprocessor.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def map_to_latent_space(model, tensor_data, batch_size=256):
    """
    Autoencoder를 사용하여 데이터를 Latent Space로 맵핑합니다.
    Args:
        model: 학습된 Autoencoder 모델
        tensor_data: 전처리된 데이터 (텐서 딕셔너리 형태)
        batch_size: 배치 크기
    Returns:
        Latent Space 표현 (텐서)
    """
    dataset = CategoricalDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    latent_spaces = []
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            encoded, _ = model(batch_data)
            latent_spaces.append(encoded)

    # 배치별 결과를 연결
    return torch.cat(latent_spaces, dim=0)

if __name__ == '__main__':
    preprocessor = joblib.load(preprocessor_path)
    model = CategoricalAutoencoder(
        categorical_dims=preprocessor.categorical_dims,  # Preprocessor에서 다시 설정됨
        preprocessor=preprocessor,
        embedding_dims=None,    # Preprocessor에서 다시 설정됨
        hidden_dim=128
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    data_path = './B2B/train_df.csv'  # 테스트 데이터 경로
    df = pd.read_csv(data_path, encoding='utf-8')

    object_df = df.select_dtypes(include=['object'])
    object_df.fillna('Missing', inplace=True)
    object_df.drop(columns=['not_converted_reason'], inplace=True)
    if 'lead_status' in object_df.columns:
        object_df.drop(columns=['lead_status'], inplace=True)
    tensor_data = preprocessor.transform(object_df)

    latent_space = map_to_latent_space(model, tensor_data)
    latent_space_df = pd.DataFrame(latent_space.cpu().numpy())
    latent_space_df.to_csv('./latent_space_results.csv', index=False)

    print("Latent Space results saved to './latent_space_results.csv'")
    exit()