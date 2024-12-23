from torch.utils.data import Dataset
# Custom Dataset 클래스 정의
class CategoricalDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): 텐서 형태의 데이터 딕셔너리
        """
        self.data_dict = data_dict
        self.length = len(next(iter(data_dict.values())))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data_dict.items()}