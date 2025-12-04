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
                'text': data['sentence']
            })
    return sentences

def get_outer_inner_entities(entities):
    """
    辅助函数：获取最外层和最内层实体。
    * 简化处理：这里假设只有两层嵌套。
    """
    if not entities:
        return set(), set()
    
    # 1. 找到所有跨度的边界 (start, end)
    spans = sorted(list(set([(e[0], e[1]) for e in entities])))
    
    outer_entities = set()
    inner_entities = set()
    
    for i, (s1, e1, t1) in enumerate(entities):
        is_nested = False
        is_outer = True
        
        for j, (s2, e2, t2) in enumerate(entities):
            if i == j:
                continue
            
            # 检查包含关系
            if s1 >= s2 and e1 <= e2 and (s1 != s2 or e1 != e2):
                # (s1, e1) 被 (s2, e2) 包含，则 (s1, e1) 可能是内层实体
                is_nested = True
                
            if s2 >= s1 and e2 <= e1 and (s1 != s2 or e1 != e2):
                # (s1, e1) 包含 (s2, e2)，则 (s1, e1) 可能是外层实体，(s2, e2) 是内层实体
                is_outer = False

        entity_tuple = (s1, e1, t1)
        if is_outer:
            # 如果它不被任何其他实体包含，则认为是外层实体
            outer_entities.add(entity_tuple)
        elif is_nested and not any(e for e in entities if e[0] < s1 and e[1] > e1):
            # 如果它被嵌套，且不是最内层（简化逻辑：这里难以精确判断“最内层”），
            # 为了实现“内外层”识别，我们只取最大不重叠实体作为外层，被包含的作为内层。
            # 实际实现中，通常会使用更精确的深度/层级信息。
            # 这里简化为：如果它被嵌套，且没有其他实体比它更“内”（即包含它），则认为是内层。
            pass # 进一步简化，只使用BIO标签来区分最外层和非最外层

    # 简单实现：只将最长（非重叠）实体作为"OUTER"，其余为"INNER" (用于Layering方法 1)
    # 这不是标准的做法，但演示了分层思想。
    max_span_length = max([e[1] - e[0] for e in entities]) if entities else 0
    outer_entities = set(e for e in entities if e[1] - e[0] == max_span_length)
    inner_entities = set(e for e in entities if e not in outer_entities)
    
    return outer_entities, inner_entities

def convert_to_bio(tokens, entities, entity_type=None, is_outer_layer=False):
    """将实体列表转换为BIO标签序列，支持选择特定类型或层级。"""
    bio_tags = ["O"] * len(tokens)
    target_tags = ENTITY_TYPES if entity_type is None else [entity_type]
    
    for start, end, etype in entities:
        if etype in target_tags:
            # 在嵌套场景中，简化后的BIO标签
            tag_prefix = "B-" if is_outer_layer else "I-"  # 简化标签，实际中标签名需区分类型
            
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

        # 1. Layering Method (内外层 BIO)
        outer_entities, inner_entities = get_outer_inner_entities(all_entities)
        
        # 实际操作中，通常为每个实体类型训练一个模型（即Cascading），或者使用专门的编码方案。
        # 这里为了模拟"Layering"方法，我们创建两个简化的标签序列：
        # Layer 1: Outermost entities only (Simplified by length for demonstration)
        outer_tags = convert_to_bio(tokens, outer_entities, is_outer_layer=True)
        # Layer 2: Innermost entities only (Simplified by exclusion for demonstration)
        inner_tags = convert_to_bio(tokens, inner_entities, is_outer_layer=False)
        
        processed_data['layering'].append({
            'tokens': tokens,
            'outer_tags': outer_tags,
            'inner_tags': inner_tags
        })

        # 2. Cascading Method (每个实体类型一个独立的BIO序列)
        cascading_item = {'tokens': tokens}
        for etype in ENTITY_TYPES:
            # 筛选当前实体类型的所有实体
            type_entities = [e for e in all_entities if e[2] == etype]
            type_tags = convert_to_bio(tokens, type_entities, entity_type=etype)
            cascading_item[f'{etype}_tags'] = type_tags
        processed_data['cascading'].append(cascading_item)

        # 3. Enumeration/Span-Based (所有可能的实体跨度列表)
        # Span-Based方法需要所有（或采样过的）(start, end)对和它们对应的标签
        span_labels = {}
        for s, e, t in all_entities:
            span_labels[(s, e)] = t
        
        all_spans = []
        MAX_SPAN_LEN = 10  # 限制最大跨度长度以减少计算量
        
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens) + 1, i + MAX_SPAN_LEN)):
                span = (i, j)
                label = span_labels.get(span, "O") # "O"表示非实体
                all_spans.append({'span': span, 'label': label})
        
        processed_data['span_based'].append({
            'tokens': tokens,
            'spans': all_spans
        })
        
    # 4. ReasoningIE (与Span-Based类似，但通常需要更复杂的编码)
    # 沿用 Span-Based 的数据格式，模型中实现复杂逻辑
    processed_data['reasoning_ie'] = processed_data['span_based'] 

    return processed_data


all_data = load_data("data/train.jsonlines")
processed_data = preprocess_for_all_methods(all_data)
print(f"加载 {len(all_data)} 条数据，准备进行四种方法训练。")