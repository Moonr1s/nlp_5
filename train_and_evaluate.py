import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
import numpy as np
import os

# 导入所有常量和模型
# 确保 constants.py 和 models.py 在同一目录下
from constants import TAG_TO_ID, SPAN_LABEL_TO_ID, ENTITY_TYPES, MODEL_NAME
from models import LayeringNERModel, SingleTypeNERModel, SpanBasedNERModel, ReasoningIENERModel

# 导入数据处理模块
# 确保 data_processor.py 在同一目录下
from data_processor import load_data, preprocess_for_all_methods

# --- 辅助函数：将标签字符串转换为 ID ---
def tags_to_ids(tags, tag_map):
    return [tag_map.get(t, -100) for t in tags]

# --- 通用 Data Collator (针对序列标注) ---
def data_collator_sequence(batch, tokenizer, tag_map, is_layering=False):
    tokens = [item['tokens'] for item in batch]
    
    # 1. Tokenization
    encoded_inputs = tokenizer(tokens, 
                               is_split_into_words=True, 
                               padding='max_length', 
                               truncation=True, 
                               return_tensors='pt')
    
    batch_labels_outer = []
    batch_labels_inner = []
    
    for i, item in enumerate(batch):
        word_ids = encoded_inputs.word_ids(batch_index=i)
        
        # 获取标签序列
        if is_layering:
            raw_tags_outer = item['outer_tags']
            raw_tags_inner = item['inner_tags']
        else:
            # Cascading 模式下，这里简化处理 PER 类型，实际应根据模型类型传入参数
            etype = 'PER' 
            # 注意：如果使用 Cascading，需要确保 data_processor 生成了对应的 tags
            raw_tags_outer = item.get(f'{etype}_tags', ['O'] * len(item['tokens']))
            raw_tags_inner = [t.replace(f'B-{etype}', 'B').replace(f'I-{etype}', 'I') 
                              for t in raw_tags_outer] 
            
        previous_word_idx = None
        label_ids_outer = []
        label_ids_inner = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)
            elif word_idx != previous_word_idx:
                # 确保索引不越界 (以防 tokenization 和原始 tokens 不对齐的情况)
                if word_idx < len(raw_tags_outer):
                    tag_outer = raw_tags_outer[word_idx]
                    label_ids_outer.append(tag_map.get(tag_outer, -100))
                    
                    tag_inner = raw_tags_inner[word_idx]
                    # Cascading/Single-Type 模型映射
                    tag_map_single = {'O': 0, f'B-{etype}': 1, f'I-{etype}': 2}
                    label_ids_inner.append(tag_map_single.get(tag_inner, -100) if not is_layering else tag_map.get(tag_inner, -100))
                else:
                    label_ids_outer.append(-100)
                    label_ids_inner.append(-100)
            else:
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)

            previous_word_idx = word_idx

        batch_labels_outer.append(label_ids_outer)
        batch_labels_inner.append(label_ids_inner)

    if is_layering:
        return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(batch_labels_outer), torch.tensor(batch_labels_inner)
    else:
        return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(batch_labels_inner) 

# --- Span-Based Data Collator ---
def data_collator_span_based(batch, tokenizer, span_label_map, max_spans=100):
    if len(batch) > 1:
        raise ValueError("Span-Based 模型的 Data Collator 演示仅支持 batch_size=1。")
        
    item = batch[0]
    tokens = item['tokens']
    
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    
    all_spans = item['spans']
    selected_spans = all_spans[:max_spans] 
    
    # +1 是因为 BERT [CLS] token
    start_indices = [s['span'][0] + 1 for s in selected_spans] 
    end_indices = [s['span'][1] for s in selected_spans] # span 结束索引通常是 inclusive 或 exclusive，需根据 span 定义调整
    
    span_widths = [s['span'][1] - s['span'][0] for s in selected_spans]
    span_labels_ids = [span_label_map.get(s['label'], 0) for s in selected_spans]
    
    start_indices = torch.tensor(start_indices, dtype=torch.long)
    end_indices = torch.tensor(end_indices, dtype=torch.long)
    span_widths = torch.tensor(span_widths, dtype=torch.long)
    span_labels_ids = torch.tensor(span_labels_ids, dtype=torch.long)
    
    return input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels_ids

# --- 真实数据集类 ---
class NERDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

# --- 训练与评估逻辑 ---
def train_and_evaluate(method_name, model, dataset, data_collator_fn, tag_map, num_epochs=3, learning_rate=1e-5):
    print(f"\n--- 启动 {method_name} 方法训练 ---")
    print(f"数据集大小: {len(dataset)}")
    
    # 检查是否有 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 根据方法选择 Collator 和 Batch Size
    if "Layering" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=True)
        batch_size = 4
    elif "Cascading" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=False)
        batch_size = 4
    elif "Span-Based" in method_name or "ReasoningIE" in method_name:
        collator = lambda batch: data_collator_span_based(batch, tokenizer, tag_map)
        batch_size = 1 
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(dataloader):
            # 将数据移动到 GPU
            batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            
            optimizer.zero_grad()
            
            if "Layering" in method_name:
                input_ids, attention_mask, labels_outer, labels_inner = batch
                loss, _, _ = model(input_ids, attention_mask, labels_outer, labels_inner)
            
            elif "Cascading" in method_name:
                input_ids, attention_mask, labels = batch 
                loss, _ = model(input_ids, attention_mask, labels)
                
            elif "Span-Based" in method_name or "ReasoningIE" in method_name:
                input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels = batch
                loss, _ = model(input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                 print(f"  Batch {i+1}/{len(dataloader)} Loss: {total_loss / (i+1):.4f}")

    print(f"训练完成：{method_name}。")

# --- 主实验函数 ---
def run_experiment_pipeline(data_files):
    # 获取训练数据路径
    train_file = data_files[0]
    
    if not os.path.exists(train_file):
        print(f"错误: 找不到文件 {train_file}")
        print("请确保数据文件位于 'data/train.jsonlines' 或修改路径。")
        return

    print(f"正在加载数据: {train_file} ...")
    raw_data = load_data(train_file)
    
    print("正在预处理数据 (生成 Layering, Cascading, Span-based 格式)...")
    processed_data = preprocess_for_all_methods(raw_data)
    
    # 2. 模型定义和实验运行
    
    # --- 方法 1: Layering Method ---
    layering_model = LayeringNERModel()
    layering_dataset = NERDataset(processed_data['layering'])
    
    print("\n------------------------------------------------------------")
    print(f"开始运行 Layering Method 训练 (数据量: {len(layering_dataset)})")
    print("------------------------------------------------------------")
    
    # 运行 Layering Method
    # data_collator_fn 参数在这里传入 None，因为我们在 train_and_evaluate 内部根据方法名选择了 collator
    train_and_evaluate("Layering Method", layering_model, layering_dataset, None, TAG_TO_ID)

    #--- 方法 2: Cascading Method (以 PER 为例) ---
    cascading_per_model = SingleTypeNERModel('PER')
    cascading_dataset = NERDataset(processed_data['cascading']) # 注意: 这里需要筛选或调整 dataset 以仅包含有 PER 标签的数据
    train_and_evaluate("Cascading Method (PER)", cascading_per_model, cascading_dataset, None, TAG_TO_ID)

    #--- 方法 3: Span-Based Method ---
    span_model = SpanBasedNERModel()
    span_dataset = NERDataset(processed_data['span_based'])
    train_and_evaluate("Span-Based Method", span_model, span_dataset, None, SPAN_LABEL_TO_ID)


if __name__ == '__main__':  
    # 假设数据在当前目录下的 data 文件夹中
    data_path = os.path.join("data", "train.jsonlines")
    run_experiment_pipeline([data_path])