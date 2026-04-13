# Weibo User Type Classification: Hierarchical User Profiling with RoBERTa-wwm-ext-large

[**中文说明**](./README_zh.md) | [**English**](./README_en.md)

> This project builds a fine-grained user classification system based on Weibo user profile information. It uses **RoBERTa-wwm-ext-large** for Chinese pre-trained language model fine-tuning, combined with a **hierarchical routing strategy**, **label restructuring**, and **training strategy optimization** to improve multi-class classification performance.

---

## Overview

This project focuses on fine-grained Weibo user type classification using user profile information from Weibo.  
The original task contains **24 user categories**, including government-related accounts, media, schools, enterprises, social organizations, self-media accounts, and ordinary internet users.

The core work of this project includes:

- Fine-tuning a **Chinese pre-trained language model** based on **RoBERTa-wwm-ext-large**
- Designing a staged training pipeline for **class imbalance**
- Building a **hierarchical routing classification framework** based on class separability
- Improving model performance through **label cleaning**, **input normalization**, **rule-based filtering**, and **iterative experiments**

This is not just a simple text classification experiment, but a complete **algorithm engineering workflow** for task decomposition and model optimization.

---

## Highlights

- **Pre-trained model fine-tuning**: Built on [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm) for Chinese user classification.
- **Hierarchical classification framework**: Instead of directly training a single 22/24-class classifier, the task is decomposed into multiple routing stages.
- **Label restructuring**: Extremely low-frequency labels are removed, reducing the original 24-class problem into a cleaner 22-class task.
- **Imbalanced data handling**: Large and small classes are separated to reduce dominance from head classes during training.
- **Training strategy optimization**: Experiments show that unified text length preprocessing noticeably improves classification performance.
- **Engineering-oriented iteration**: The project emphasizes configuration control, staged checkpointing, and progressive error analysis.

---

## Task Definition

### Input Features

Available user information includes, but is not limited to:

- Nickname
- Following count
- Follower count
- Weibo post count
- Verification information
- Blogger tags
- User description
- Work information
- Labels / additional metadata

The available features include:

- **3 numerical features**
- **6 textual features**

### Original Label Set

The raw dataset contains **24 classes**:

```python
{
    '超话粉丝大咖': 0, '公务员': 1, '大V名人': 2, '党委': 3, '国防军委': 4,
    '基层组织': 5, '政府': 6, '检验检测': 7, '媒体': 8, '民主党派': 9,
    '明星红人': 10, '企事业单位': 11, '赛事活动': 12, '社会组织': 13,
    '社区组织': 14, '司法机关': 15, '外国政府机构': 16, '网民': 17,
    '行业专家': 18, '学校': 19, '研究机构': 20, '演艺娱乐明星': 21,
    '政协人大': 22, '自媒体': 23
}
```

---

## Dataset

- About **12,000 labeled samples**
- The dataset size is suitable for **pre-trained language model fine-tuning**
- The label distribution is highly imbalanced
- Some classes are too rare to be modeled directly in a unified classifier

---

## Methodology

Instead of using a flat one-shot classifier, this project adopts an engineering-oriented task decomposition strategy.

### 1. Label Cleaning and Restructuring

Initial experiments showed that some labels had too few samples and introduced unnecessary noise into training.  
Therefore, the following extremely rare labels were removed:

- 明星红人
- 民主党派

As a result, the original **24-class problem** was reduced to a **22-class task**.

---

### 2. Split by Class Size: Large vs. Small Classes

Since class sizes vary significantly, directly training a single classifier may bias optimization toward dominant classes while hurting minority classes.

To address this, the dataset is first split into:

#### Large Classes (12 classes)

- 社区组织
- 党委
- 自媒体
- 网民
- 媒体
- 司法机关
- 学校
- 超话粉丝大咖
- 企事业单位
- 大V名人
- 社会组织
- 政府

#### Small Classes (10 classes)

- 基层组织
- 赛事活动
- 研究机构
- 检验检测
- 政协人大
- 演艺娱乐明星
- 公务员
- 行业专家
- 外国政府机构
- 国防军委

The experiment shows that **routing samples by class size first** significantly improves downstream classification performance.

![result1. size_of_data.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result1.%20size_of_data.png?raw=true)

---

### 3. Small-Class Subtask: 10-Way Classification

For low-frequency labels, a dedicated 10-class classifier is trained:

```python
{
    '公务员': 0, '国防军委': 1, '基层组织': 2, '检验检测': 3,
    '赛事活动': 4, '外国政府机构': 5, '行业专家': 6,
    '研究机构': 7, '演艺娱乐明星': 8, '政协人大': 9
}
```

![result2. small_data_10.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result2.%20small_data_10.png?raw=true)

---

### 4. Large-Class Baseline: 12-Way Classification

A direct 12-class classifier is trained for the large-class subset.  
The results indicate that several classes are still highly confusable.

```python
{
    '超话粉丝大咖': 0, '大V名人': 1, '党委': 2, '政府': 3,
    '媒体': 4, '企事业单位': 5, '社会组织': 6, '社区组织': 7,
    '司法机关': 8, '网民': 9, '学校': 10, '自媒体': 11
}
```

![result3. big_data_12.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result3.%20big_data_12.png?raw=true)

---

### 5. Further Split by Feature-Space Separability

For the large-class subset, label distributions in feature space are further analyzed.  
Some classes exhibit strong separability and can be solved earlier, while others remain highly similar and require deeper routing.

Therefore, large classes are split into:

- **Distinctive classes**
- **Similar classes**

![result4. similar_distinctive.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result4.%20similar_distinctive.png?raw=true)

---

### 6. Distinctive-Class Classification

A dedicated classifier is trained for highly separable classes:

```python
{
    '超话粉丝大咖': 0, '政府': 1, '社会组织': 2,
    '社区组织': 3, '司法机关': 4, '学校': 5
}
```

![result5. distinctive_data.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result5.%20%20distinctive_data.png?raw=true)

---

### 7. Similar-Class Classification

A separate classifier is trained for the remaining hard-to-distinguish labels:

```python
{
    '大V名人': 0, '党委': 1, '媒体': 2,
    '企事业单位': 3, '网民': 4, '自媒体': 5
}
```

![result6. similar_6.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result6.%20%20similar_6.png?raw=true)

---

### 8. Routing Out a Key Class: 党委

During experiments, `党委` was found to be more separable than the remaining similar classes under some settings.  
Therefore, it was routed out as a dedicated binary classification task.

```python
{
    '其他': 0,
    '党委': 1
}
```

![result7. similar-first.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result7.%20%20similar-first.png?raw=true)

---

### 9. Handling a Key Confusion Pair: 网民 + 自媒体

Further analysis showed that `网民` and `自媒体` had a strong impact on overall classification quality.  
They were therefore separated into an additional routing stage.

```python
{
    '其他': 0,
    '分出网友+自媒体': 1
}
```

![result8. netzen+we-other.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result8.%20netzen+we-other.png?raw=true)

---

### 10. Final 3-Class Classifier

The final remaining difficult labels are handled by a dedicated 3-way classifier:

```python
{
    '大V名人': 0,
    '媒体': 1,
    '企事业单位': 2
}
```

![result9. last3.png](https://github.com/hawkforever5/BERT_User-Classification/blob/main/pic/result9.%20last3.png?raw=true)

---

## Key Engineering Insights

### Unified text length preprocessing matters
Experiments show that normalizing text length significantly improves classification performance.  
This suggests that input standardization helps the model learn a more stable representation space and cleaner decision boundaries.

### Class imbalance cannot be solved by a single flat classifier
For long-tail multi-class tasks, directly training one classifier often sacrifices minority-class quality for global accuracy.  
A more practical solution is **task decomposition + hierarchical routing**.

### Hierarchical routing is more suitable than flat classification for complex label systems
When the number of classes is large and inter-class similarity varies significantly, a staged pipeline is more effective than a one-shot classifier.

---

## What This Project Demonstrates

This project reflects the following algorithm engineering skills:

- Chinese NLP task modeling
- Fine-tuning large pre-trained language models
- Label engineering and task restructuring
- Imbalanced classification handling
- Hierarchical routing classifier design
- Error analysis and experiment-driven iteration
- Configuration-based experiment management

---

## Future Work

- Add more systematic evaluation metrics such as Macro-F1, per-class recall, and confusion matrix analysis
- Fuse structured numerical features with text features for a multi-modal user classification model
- Explore focal loss or class-balanced loss for long-tail learning
- Try more advanced strategies such as knowledge distillation, contrastive learning, or prompt-based fine-tuning
- Package the current pipeline into a reusable inference service or offline batch-processing workflow

---

## Reference Model

- [Chinese-BERT-wwm / RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)

---

## Notes

This repository is intended to showcase the modeling and experiment process for fine-grained Weibo user classification.  
If you are interested in the project, discussions are welcome on:

- Text classification
- Chinese PLM fine-tuning
- Hierarchical classification pipelines
- Imbalanced learning
- User profiling
