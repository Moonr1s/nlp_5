# filename: constants.py (你需要创建此文件)

# 假设的实体类型集合（基于您数据中的PER, ORG, GPE等标签）
ENTITY_TYPES = ["PER", "ORG", "GPE", "LOC", "WEA", "FAC", "VEH"]
MODEL_NAME = 'bert-base-chinese' # 假设中文数据集使用中文BERT
EMBED_DIM = 768 # BERT的输出维度
MAX_SPAN_LEN = 10 # 限制Span-Based模型最大跨度长度

# --- 序列标注 (Sequence Tagging) 标签映射 (用于 Layering / Cascading) ---
TAG_TO_ID = {"O": 0}
tag_id = 1
for etype in ENTITY_TYPES:
    TAG_TO_ID[f"B-{etype}"] = tag_id      # Begin Tag
    TAG_TO_ID[f"I-{etype}"] = tag_id + 1  # Inside Tag
    tag_id += 2
NUM_SEQ_LABELS = len(TAG_TO_ID) # 1 (O) + 2 * len(ENTITY_TYPES)

# --- 跨度分类 (Span Classification) 标签映射 (用于 Span-Based / ReasoningIE) ---
SPAN_LABEL_TO_ID = {t: i + 1 for i, t in enumerate(ENTITY_TYPES)}
SPAN_LABEL_TO_ID["O"] = 0 # 非实体跨度
NUM_SPAN_LABELS = len(SPAN_LABEL_TO_ID) # 1 (O) + len(ENTITY_TYPES)