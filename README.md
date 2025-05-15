# 房价预测系统（House Price Prediction）

基于PyTorch深度学习框架，结合多维连续数值特征与类别特征Embedding技术，构建的精准房价预测模型。项目包含从数据清洗、特征工程、模型训练、模型保存，到基于Flask的REST API预测服务及Vue3前端交互界面。

---

## 项目背景

随着城市房地产市场的复杂化，传统的简单线性模型难以准确捕捉多维度房屋特征对价格的影响。本项目利用深度学习技术，针对上海二手房数据，通过合理的特征处理和神经网络设计，实现更准确的房价预测，帮助用户科学决策。

---

## 主要功能

- **数据预处理**：解析楼层信息、户型、朝向等多样化字段，自动提取结构化特征  
- **类别特征编码**：采用`LabelEncoder`和Embedding处理类别变量，提升模型表达能力  
- **连续特征标准化**：保证特征数值稳定，便于神经网络训练  
- **模型训练**：结合连续特征与Embedding特征，训练多层神经网络进行回归预测  
- **预测服务**：基于Flask搭建API，支持前端调用，实现实时在线预测  
- **前端交互**：基于Vue3和Axios，提供用户友好的预测输入表单和结果展示  

---

## 数据说明

- 数据集：上海二手房公开数据（`data/sh_data.csv`）  
- 主要特征包括：面积（square）、楼层信息（floor_level、total_floor）、户型（room、hall、kitchen、bath）、朝向（direction_simple）、装修（decoration）、电梯情况（elevator）、产权（ownership）、总价（total_price）  
- 对部分非结构化字段如`floor`和`size`进行正则解析，提取结构化特征  

---

## 技术栈

| 组件         | 技术/库              | 说明                   |
| ------------ | -------------------- | ---------------------- |
| 数据处理     | Pandas, re           | 数据清洗与特征提取     |
| 特征编码     | scikit-learn         | LabelEncoder、StandardScaler |
| 深度学习     | PyTorch              | 模型定义与训练         |
| 模型持久化   | joblib, torch        | 保存编码器和模型权重   |
| 后端服务     | Flask, flask_cors    | 预测API与跨域支持      |
| 前端交互     | Vue3, Axios          | 用户输入与结果展示     |

---
