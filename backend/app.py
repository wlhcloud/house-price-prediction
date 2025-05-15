# app.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import pandas as pd
from flask_cors import CORS

# 初始化
app = Flask(__name__)
CORS(app)

# 加载模型依赖
scaler_X = joblib.load("./model/scaler_X.pkl")
scaler_y = joblib.load("./model/scaler_y.pkl")
encoders = joblib.load("./model/encoders.pkl")

cont_cols = ['square', 'total_floor', 'room', 'hall', 'kitchen', 'bath']
cat_cols = ['floor_level', 'direction_simple', 'decoration', 'elevator', 'ownership']
embedding_sizes = [len(encoders[col].classes_) for col in cat_cols]

# 定义模型结构
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

# 初始化模型
model = EmbeddingHouseModel(embedding_sizes, len(cont_cols))
model.load_state_dict(torch.load("./model/embedding_model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # 构造 DataFrame
    df_input = pd.DataFrame([data])

    # 编码类别特征
    for col in cat_cols:
        le = encoders[col]
        val = data[col]
        if val not in le.classes_:
            return jsonify({"message": f"未知类别值：{col} = {val}"}), 400
        df_input[col] = le.transform([val])

    # 连续和类别分离
    X_cont = scaler_X.transform(df_input[cont_cols])
    X_cat = df_input[cat_cols].values

    x_cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
    x_cat_tensor = torch.tensor(X_cat, dtype=torch.long)

    with torch.no_grad():
        pred_scaled = model(x_cont_tensor, x_cat_tensor).item()
        pred_real = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    return jsonify({"predicted_price": round(pred_real, 2)})

if __name__ == "__main__":
    app.run(debug=True)
