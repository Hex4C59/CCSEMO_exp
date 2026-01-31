# CCSEMO 5-Fold Cross-Validation 数据划分说明

本文件夹包含 CCSEMO 数据集的 5 折交叉验证划分，采用 **Speaker-Independent** 策略，确保同一说话人的所有样本只出现在一个集合中。

## 划分策略

- **Speaker-Independent**: 训练集、验证集、测试集之间无说话人重叠
- **滚动划分**: 每折的验证集成为下一折的测试集，形成循环

## 各折统计

| Fold | Split | 样本数 | 说话人数 | 性别分布 (M/F) |
|------|-------|--------|----------|----------------|
| **Fold1** | train | 4518 | 71 | 36/35 |
|           | val   | 1506 | 20 | 9/11 |
|           | test  | 1529 | 12 | 2/10 |
| **Fold2** | train | 4534 | 62 | 28/34 |
|           | val   | 1513 | 21 | 10/11 |
|           | test  | 1506 | 20 | 9/11 |
| **Fold3** | train | 4540 | 57 | 24/33 |
|           | val   | 1500 | 25 | 13/12 |
|           | test  | 1513 | 21 | 10/11 |
| **Fold4** | train | 4548 | 53 | 21/32 |
|           | val   | 1505 | 25 | 13/12 |
|           | test  | 1500 | 25 | 13/12 |
| **Fold5** | train | 4519 | 66 | 32/34 |
|           | val   | 1529 | 12 | 2/10 |
|           | test  | 1505 | 25 | 13/12 |

## 说话人划分详情

### Fold1
- **Train (71人)**: spk11, spk16, spk19, spk36, spk50, spk70, spk74, spk76, spk90, spk99, spk100, spk144, spk157, spk161, spk166, spk168, spk177, spk181, spk182, spk184, spk189, spk195, spk207, spk301, spk302, spk304, spk305, spk308, spk309, spk310, spk311, spk312, spk315, spk316, spk318, spk319, spk320, spk321, spk322, spk323, spk324, spk326, spk327, spk328, spk331, spk332, spk333, spk335, spk336, spk337, spk338, spk340, spk343, spk345, spk346, spk347, spk349, spk350, spk353, spk354, spk355, spk357, spk358, spk359, spk361, spk362, spk363, spk365, spk366, spk367, spk368
- **Val (20人)**: spk28, spk51, spk77, spk81, spk191, spk199, spk204, spk300, spk313, spk314, spk325, spk329, spk339, spk342, spk348, spk351, spk356, spk360, spk364, spk369
- **Test (12人)**: spk65, spk84, spk119, spk142, spk155, spk156, spk160, spk303, spk306, spk307, spk330, spk352

### Fold2
- **Train (62人)**: spk11, spk19, spk36, spk65, spk70, spk74, spk76, spk84, spk90, spk119, spk142, spk144, spk155, spk156, spk160, spk177, spk181, spk182, spk184, spk189, spk195, spk207, spk301, spk302, spk303, spk304, spk305, spk306, spk307, spk308, spk309, spk311, spk312, spk315, spk316, spk318, spk319, spk320, spk321, spk323, spk324, spk326, spk328, spk330, spk331, spk333, spk336, spk337, spk343, spk345, spk347, spk349, spk350, spk352, spk353, spk355, spk357, spk358, spk359, spk361, spk362, spk368
- **Val (21人)**: spk16, spk50, spk99, spk100, spk157, spk161, spk166, spk168, spk310, spk322, spk327, spk332, spk335, spk338, spk340, spk346, spk354, spk363, spk365, spk366, spk367
- **Test (20人)**: spk28, spk51, spk77, spk81, spk191, spk199, spk204, spk300, spk313, spk314, spk325, spk329, spk339, spk342, spk348, spk351, spk356, spk360, spk364, spk369

### Fold3
- **Train (57人)**: spk19, spk28, spk51, spk65, spk70, spk74, spk76, spk77, spk81, spk84, spk90, spk119, spk142, spk155, spk156, spk160, spk182, spk184, spk191, spk195, spk199, spk204, spk300, spk301, spk302, spk303, spk306, spk307, spk308, spk313, spk314, spk316, spk318, spk319, spk325, spk326, spk328, spk329, spk330, spk331, spk336, spk339, spk342, spk345, spk348, spk349, spk350, spk351, spk352, spk353, spk356, spk358, spk359, spk360, spk361, spk364, spk369
- **Val (25人)**: spk11, spk36, spk144, spk177, spk181, spk189, spk207, spk304, spk305, spk309, spk311, spk312, spk315, spk320, spk321, spk323, spk324, spk333, spk337, spk343, spk347, spk355, spk357, spk362, spk368
- **Test (21人)**: spk16, spk50, spk99, spk100, spk157, spk161, spk166, spk168, spk310, spk322, spk327, spk332, spk335, spk338, spk340, spk346, spk354, spk363, spk365, spk366, spk367

### Fold4
- **Train (53人)**: spk16, spk28, spk50, spk51, spk65, spk77, spk81, spk84, spk99, spk100, spk119, spk142, spk155, spk156, spk157, spk160, spk161, spk166, spk168, spk191, spk199, spk204, spk300, spk303, spk306, spk307, spk310, spk313, spk314, spk322, spk325, spk327, spk329, spk330, spk332, spk335, spk338, spk339, spk340, spk342, spk346, spk348, spk351, spk352, spk354, spk356, spk360, spk363, spk364, spk365, spk366, spk367, spk369
- **Val (25人)**: spk19, spk70, spk74, spk76, spk90, spk182, spk184, spk195, spk301, spk302, spk308, spk316, spk318, spk319, spk326, spk328, spk331, spk336, spk345, spk349, spk350, spk353, spk358, spk359, spk361
- **Test (25人)**: spk11, spk36, spk144, spk177, spk181, spk189, spk207, spk304, spk305, spk309, spk311, spk312, spk315, spk320, spk321, spk323, spk324, spk333, spk337, spk343, spk347, spk355, spk357, spk362, spk368

### Fold5
- **Train (66人)**: spk11, spk16, spk28, spk36, spk50, spk51, spk77, spk81, spk99, spk100, spk144, spk157, spk161, spk166, spk168, spk177, spk181, spk189, spk191, spk199, spk204, spk207, spk300, spk304, spk305, spk309, spk310, spk311, spk312, spk313, spk314, spk315, spk320, spk321, spk322, spk323, spk324, spk325, spk327, spk329, spk332, spk333, spk335, spk337, spk338, spk339, spk340, spk342, spk343, spk346, spk347, spk348, spk351, spk354, spk355, spk356, spk357, spk360, spk362, spk363, spk364, spk365, spk366, spk367, spk368, spk369
- **Val (12人)**: spk65, spk84, spk119, spk142, spk155, spk156, spk160, spk303, spk306, spk307, spk330, spk352
- **Test (25人)**: spk19, spk70, spk74, spk76, spk90, spk182, spk184, spk195, spk301, spk302, spk308, spk316, spk318, spk319, spk326, spk328, spk331, spk336, spk345, spk349, spk350, spk353, spk358, spk359, spk361

## 滚动关系

```
Fold1: val  → Fold2: test
Fold2: val  → Fold3: test
Fold3: val  → Fold4: test
Fold4: val  → Fold5: test
Fold5: val  → Fold1: test
```

## 文件格式

每个 `foldX.csv` 包含以下列：
- `audio_path`: 音频文件路径
- `name`: 文件名
- `V`: Valence 值
- `A`: Arousal 值
- `gender`: 性别 (M/F)
- `duration`: 时长（秒）
- `discrete_emotion`: 离散情感标签
- `split_set`: 数据集划分 (train/val/test)
- `transcript`: 转录文本
