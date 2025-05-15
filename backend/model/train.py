# train_model.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import re

# 1. 数据加载和预处理
df = pd.read_csv("../../data/sh_data.csv")

def extract_floor_info(floor_str):
    match = re.search(r'(低|中|高)楼层\s*\(共(\d+)层\)', str(floor_str))
    return (match.group(1), int(match.group(2))) if match else ('未知', 0)

def parse_size(size_str):
    r = lambda p: int(re.search(p, str(size_str)).group(1)) if re.search(p, str(size_str)) else 0
    return [r(r'(\d+)室'), r(r'(\d+)厅'), r(r'(\d+)厨'), r(r'(\d+)卫')]

df[['floor_level', 'total_floor']] = df['floor'].apply(lambda x: pd.Series(extract_floor_info(x)))
df[['room', 'hall', 'kitchen', 'bath']] = df['size'].apply(lambda x: pd.Series(parse_size(x)))
df['direction_simple'] = df['direction'].str.replace(' ', '')

features = ['square', 'floor_level', 'total_floor', 'room', 'hall', 'kitchen', 'bath',
            'direction_simple', 'decoration', 'elevator', 'ownership', 'total_price']
df = df[features].dropna()

# 连续与类别特征
cont_cols = ['square', 'total_floor', 'room', 'hall', 'kitchen', 'bath']
cat_cols = ['floor_level', 'direction_simple', 'decoration', 'elevator', 'ownership']

# 编码类别特征
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 保存编码器
joblib.dump(encoders, "encoders.pkl")

# 标准化连续特征
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_cont = scaler_X.fit_transform(df[cont_cols])
y = scaler_y.fit_transform(df[['total_price']])
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# 构造训练数据
X_cat = df[cat_cols].values
X_cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float32)

Xc_train, Xc_test, Xcat_train, Xcat_test, y_train, y_test = train_test_split(
    X_cont_tensor, X_cat_tensor, y_tensor, test_size=0.2, random_state=42)

# 2. 模型定义
class EmbeddingHouseModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, 8) for cat_size in embedding_sizes])
        self.fc = nn.Sequential(
            nn.Linear(n_cont + 8 * len(embedding_sizes), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cont, x_cat):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x + [x_cont], dim=1)
        return self.fc(x)

embedding_sizes = [len(encoders[col].classes_) for col in cat_cols]
model = EmbeddingHouseModel(embedding_sizes, len(cont_cols))

# 3. 模型训练
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(Xc_train, Xcat_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. 保存模型
torch.save(model.state_dict(), "embedding_model.pth")
print("✅ 模型和编码器已保存")