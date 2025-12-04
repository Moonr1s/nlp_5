import json
from collections import defaultdict

# 假设的实体类型集合（基于您数据中的PER, ORG, GPE等标签）
ENTITY_TYPES = ["PER", "ORG", "GPE", "LOC", "WEA", "FAC", "VEH"]
INNER_OUTER_TAGS = ["B-OUTER", "I-OUTER", "B-INNER", "I-INNER", "O"]
CASCADING_TAGS = {etype: [f"B-{etype}", f"I-{etype}", "O"] for etype in ENTITY_TYPES}
CASCADING_MODELS = ENTITY_TYPES

def load_data(file_path):
    """加载原始的JSONLines数据，并提取tokens和所有实体提及。"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 标记实体为 (start, end, type) 格式，其中end是排他的 (exclusive)
            entities = [(e['start'], e['end'], e['entity_type']) 
                        for e in data['entity_mentions']]
            sentences.append({
                'tokens': data['tokens'],
                'entities': entities,
                'text': data.get('sentence', "".join(data['tokens'])) # 获取或还原文本
            })
    return sentences

def get_outer_inner_entities(entities):
    """辅助函数：获取最外层和最内层实体 (简化版)。"""
    if not entities:
        return set(), set()
    
    max_span_length = max([e[1] - e[0] for e in entities]) if entities else 0
    # 简单规则：最长的为 Outer，其余为 Inner
    outer_entities = set(e for e in entities if e[1] - e[0] == max_span_length)
    inner_entities = set(e for e in entities if e not in outer_entities)
    
    return outer_entities, inner_entities

def convert_to_bio(tokens, entities, entity_type=None, is_outer_layer=False):
    """将实体列表转换为BIO标签序列。"""
    bio_tags = ["O"] * len(tokens)
    target_tags = ENTITY_TYPES if entity_type is None else [entity_type]
    
    for start, end, etype in entities:
        if etype in target_tags:
            tag_prefix = "B-" if is_outer_layer else "I-" # 简化
            if start < len(tokens):
                bio_tags[start] = f"B-{etype}"
            for i in range(start + 1, end):
                if i < len(tokens):
                    bio_tags[i] = f"I-{etype}"
    return bio_tags

def preprocess_for_all_methods(data):
    """为所有四种方法生成所需的数据格式。"""
    processed_data = defaultdict(list)

    for item in data:
        tokens = item['tokens']
        all_entities = item['entities']
        text = item['text']

        # --- 1. Layering Method (内外层 BIO) ---
        outer_entities, inner_entities = get_outer_inner_entities(all_entities)
        processed_data['layering'].append({
            'tokens': tokens,
            'outer_tags': convert_to_bio(tokens, outer_entities, is_outer_layer=True),
            'inner_tags': convert_to_bio(tokens, inner_entities, is_outer_layer=False)
        })

        # --- 2. Cascading Method (每个类型独立) ---
        cascading_item = {'tokens': tokens}
        for etype in ENTITY_TYPES:
            type_entities = [e for e in all_entities if e[2] == etype]
            cascading_item[f'{etype}_tags'] = convert_to_bio(tokens, type_entities, entity_type=etype)
        processed_data['cascading'].append(cascading_item)

        # --- 3. Span-Based Method (跨度枚举) ---
        span_labels = {}
        for s, e, t in all_entities:
            span_labels[(s, e)] = t
        
        all_spans = []
        MAX_SPAN_LEN = 10 
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens) + 1, i + MAX_SPAN_LEN)):
                span = (i, j)
                label = span_labels.get(span, "O")
                all_spans.append({'span': span, 'label': label})
        
        processed_data['span_based'].append({
            'tokens': tokens,
            'spans': all_spans
        })
        
        # --- 4. ReasoningIE (Generative / Seq2Seq) ---
        # 目标：生成 "TYPE: mention; TYPE: mention" 格式的字符串
        # 按照在文本中出现的顺序排序实体
        sorted_entities = sorted(all_entities, key=lambda x: x[0])
        target_parts = []
        for s, e, t in sorted_entities:
            # 从 tokens 还原实体文本
            entity_text = "".join(tokens[s:e])
            target_parts.append(f"{t}: {entity_text}")
        
        target_text = "; ".join(target_parts) if target_parts else "无实体"
        
        processed_data['reasoning_ie'].append({
            'input_text': text,         # 输入：原始句子
            'target_text': target_text  # 输出：结构化字符串
        })

    return processed_data