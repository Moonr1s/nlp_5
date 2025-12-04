# filename: train_and_evaluate.py (关键更新部分)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW # 导入 AdamW 优化器
import numpy as np

# 导入所有常量和模型
from constants import TAG_TO_ID, SPAN_LABEL_TO_ID, ENTITY_TYPES
from models import LayeringNERModel, SingleTypeNERModel, SpanBasedNERModel, ReasoningIENERModel, MODEL_NAME

# --- 辅助函数：将标签字符串转换为 ID ---
def tags_to_ids(tags, tag_map):
    # 将列表中的标签字符串映射为 ID，对于不存在的标签 ID 使用 -100 (用于 CrossEntropyLoss 忽略)
    return [tag_map.get(t, -100) for t in tags]

# --- 通用 Data Collator (针对序列标注) ---
def data_collator_sequence(batch, tokenizer, tag_map, is_layering=False):
    tokens = [item['tokens'] for item in batch]
    
    # 1. Tokenization (使用 BERT tokenizer)
    encoded_inputs = tokenizer(tokens, 
                               is_split_into_words=True, 
                               padding='max_length', 
                               truncation=True, 
                               return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    
    batch_labels_outer = []
    batch_labels_inner = []
    
    for i, item in enumerate(batch):
        # 2. 标签对齐到 Sub-word 级别
        word_ids = encoded_inputs.word_ids(batch_index=i)
        
        # 初始标签序列（Layering 或 Cascading）
        if is_layering:
            raw_tags_outer = item['outer_tags']
            raw_tags_inner = item['inner_tags']
        else:
            # 假设 Cascading 模型批次处理的是某个特定实体的标签
            # 简化：这里需要用户传入要处理的实体类型
            etype = 'PER' # 假设当前批次用于训练 PER 模型
            raw_tags_outer = item[f'{etype}_tags']
            raw_tags_inner = [t.replace(f'B-{etype}', 'B').replace(f'I-{etype}', 'I') 
                              for t in raw_tags_outer] # Cascading 仅需 B/I/O
            
        # 3. 标签对齐 (只对第一个 Sub-word 赋予标签，其他 Sub-word 使用 -100 忽略)
        previous_word_idx = None
        label_ids_outer = []
        label_ids_inner = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], Padding token 
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)
            elif word_idx != previous_word_idx:
                # 句子中的第一个 sub-word 标记实际标签
                tag_outer = raw_tags_outer[word_idx]
                label_ids_outer.append(tag_map.get(tag_outer, -100))
                
                tag_inner = raw_tags_inner[word_idx]
                # Cascading/Single-Type 模型需要 B/I/O 映射 (简化为 0, 1, 2)
                tag_map_single = {'O': 0, f'B-{etype}': 1, f'I-{etype}': 2}
                label_ids_inner.append(tag_map_single.get(tag_inner, -100) if not is_layering else tag_map.get(tag_inner, -100))

            else:
                # 同一个词的后续 sub-word 忽略损失
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)

            previous_word_idx = word_idx

        batch_labels_outer.append(label_ids_outer)
        batch_labels_inner.append(label_ids_inner)

    if is_layering:
        return encoded_inputs['input_ids'], attention_mask, torch.tensor(batch_labels_outer), torch.tensor(batch_labels_inner)
    else:
        # Cascading 只返回一个标签集 (简化)
        return encoded_inputs['input_ids'], attention_mask, torch.tensor(batch_labels_inner) 
    
# --- Span-Based Data Collator ---
def data_collator_span_based(batch, tokenizer, span_label_map, max_spans=100):
    
    # 简化：仅处理 batch_size=1 的情况，Span-Based 模型的批处理实现非常复杂
    if len(batch) > 1:
        # 在实际实验中，你需要实现复杂的动态 Padding 和 Span 统一化
        raise ValueError("Span-Based 模型的 Data Collator 仅支持 batch_size=1 演示。")
        
    item = batch[0]
    tokens = item['tokens']
    
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    
    # 提取 Span 数据
    all_spans = item['spans']
    
    # 仅使用前 max_spans 个 span，确保演示简洁
    selected_spans = all_spans[:max_spans] 
    
    start_indices = [s['span'][0] + 1 for s in selected_spans] # +1 偏移量用于 BERT 的 [CLS]
    end_indices = [s['span'][1] for s in selected_spans]
    
    span_widths = [s['span'][1] - s['span'][0] for s in selected_spans]
    span_labels_ids = [span_label_map.get(s['label'], 0) for s in selected_spans]
    
    # 将列表转换为 Tensor
    start_indices = torch.tensor(start_indices, dtype=torch.long)
    end_indices = torch.tensor(end_indices, dtype=torch.long)
    span_widths = torch.tensor(span_widths, dtype=torch.long)
    span_labels_ids = torch.tensor(span_labels_ids, dtype=torch.long)
    
    return input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels_ids

# --- 训练与评估逻辑 ---

def train_and_evaluate(method_name, model, dataset, data_collator_fn, tag_map, num_epochs=3, learning_rate=1e-5):
    print(f"\n--- 启动 {method_name} 方法训练 ---")
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 根据模型方法选择 Collator
    if "Layering" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=True)
        batch_size = 4
    elif "Cascading" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=False)
        batch_size = 4
    elif "Span-Based" in method_name or "ReasoningIE" in method_name:
        collator = lambda batch: data_collator_span_based(batch, tokenizer, tag_map)
        # Span-Based 演示只能用 batch_size=1
        batch_size = 1 
    else:
        raise ValueError("Unknown method name")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    # 模拟训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 使用进度条（省略实际实现）
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            if "Layering" in method_name:
                input_ids, attention_mask, labels_outer, labels_inner = batch
                loss, _, _ = model(input_ids, attention_mask, labels_outer, labels_inner)
            
            elif "Cascading" in method_name:
                input_ids, attention_mask, labels = batch # labels 是单个实体的标签
                loss, _ = model(input_ids, attention_mask, labels)
                
            elif "Span-Based" in method_name or "ReasoningIE" in method_name:
                # Span-Based/ReasoningIE 的 forward 接收 6 个参数
                input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels = batch
                loss, _ = model(input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels)
                
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                 print(f"  Batch {i+1}/{len(dataloader)} Loss: {total_loss / (i+1):.4f}")

    print(f"训练完成：{method_name}。")
    # 模拟评估结果
    # ... (评估逻辑省略)
    # print(f"*** 模拟评估结果： F1 Score: {0.75 + torch.randn(1).item() * 0.05:.4f} ***")

# --- 主实验函数 (需要修改以调用新的 train_and_evaluate) ---

def run_experiment_pipeline(data_files):
    # 1. 数据加载与预处理 (假设 data_processor.py 已实现 load_data 和 preprocess_for_all_methods)
    # raw_data = load_data(data_files[0]) 
    # processed_data = preprocess_for_all_methods(raw_data)
    
    # 🚨 由于无法访问原始文件，这里需要模拟 processed_data 以便演示模型初始化
    class DummyDataset(Dataset):
        def __init__(self, size):
            self.data = [{'tokens': ['这', '是', '一', '个', '示', '例'], 
                          'outer_tags': ['B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'O'],
                          'inner_tags': ['B-PER', 'I-PER', 'O', 'O', 'O', 'O'],
                          'PER_tags': ['O', 'O', 'O', 'B-PER', 'I-PER', 'O'], # Cascading
                          'spans': [{'span': (0, 2), 'label': 'ORG'}, {'span': (3, 5), 'label': 'PER'}]
                          } for _ in range(size)]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]
        
    print("使用模拟数据进行模型初始化和训练逻辑演示...")
    dummy_size = 50
    
    # 2. 模型定义和实验运行
    
    # --- 方法 1: Layering Method ---
    layering_model = LayeringNERModel()
    layering_dataset = DummyDataset(dummy_size)
    # train_and_evaluate("Layering Method (Outer/Inner BIO)", layering_model, layering_dataset, data_collator_sequence, TAG_TO_ID)
    
    # --- 方法 2: Cascading Method (以 PER 实体为例) ---
    # Cascading 模型的标签映射是精简的 (O/B-PER/I-PER)
    cascading_per_model = SingleTypeNERModel('PER')
    cascading_dataset = DummyDataset(dummy_size)
    # train_and_evaluate("Cascading Method (PER Entity)", cascading_per_model, cascading_dataset, data_collator_sequence, {'O':0, 'B-PER':1, 'I-PER':2})

    # --- 方法 3: Span-Based Method ---
    span_model = SpanBasedNERModel()
    span_dataset = DummyDataset(dummy_size)
    # train_and_evaluate("Enumeration/Span-Based Method", span_model, span_dataset, data_collator_span_based, SPAN_LABEL_TO_ID)

    # --- 方法 4: ReasoningIE Style ---
    reasoning_model = ReasoningIENERModel()
    reasoning_dataset = DummyDataset(dummy_size)
    
    print("\n------------------------------------------------------------")
    print("模型填充完成。以下是运行 Layering Method 的模拟训练演示：")
    print("------------------------------------------------------------")
    
    # 实际运行 Layering Method 演示
    train_and_evaluate("Layering Method (Outer/Inner BIO)", layering_model, layering_dataset, None, TAG_TO_ID)


if __name__ == '__main__':  
    run_experiment_pipeline(["dummy_file.jsonlines"]) # 使用模拟数据运行