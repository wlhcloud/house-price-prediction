# 房价预测系统（House Price Prediction）

基于PyTorch的多特征房价预测模型，结合连续数值特征和类别特征的Embedding技术，实现精准的房价估计。包含完整的数据预处理、模型训练脚本和Flask预测API，配套Vue前端界面。

---

## 功能介绍

- 数据预处理：对楼层、户型、朝向、装修、电梯、产权等复杂房屋信息进行解析和编码  
- 模型训练：使用连续特征和类别特征的Embedding结合的神经网络，提升预测效果  
- 预测API：基于Flask实现RESTful接口，支持实时房价预测  
- 前端页面：Vue3实现用户友好的表单输入和预测结果展示  

---

## 技术栈

- Python：Pandas, scikit-learn, PyTorch  
- Flask：轻量级后端服务  
- Vue3 + Axios：响应式前端界面  
- Joblib：模型和编码器持久化  

---

## 快速开始

1. 克隆仓库  
```bash
git clone https://github.com/你的用户名/house-price-prediction.git
cd house-price-prediction
