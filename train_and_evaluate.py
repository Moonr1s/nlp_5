import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
import numpy as np
import os

from constants import TAG_TO_ID, SPAN_LABEL_TO_ID, ENTITY_TYPES, MODEL_NAME, MAX_GEN_LEN
from models import LayeringNERModel, SingleTypeNERModel, SpanBasedNERModel, ReasoningIENERModel
from data_processor import load_data, preprocess_for_all_methods

# --- 辅助函数：将标签字符串转换为 ID ---
def tags_to_ids(tags, tag_map):
    return [tag_map.get(t, -100) for t in tags]

# --- 1. Sequence Data Collator (Layering / Cascading) ---
def data_collator_sequence(batch, tokenizer, tag_map, is_layering=False):
    tokens = [item['tokens'] for item in batch]
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt')
    
    batch_labels_outer = []
    batch_labels_inner = []
    
    for i, item in enumerate(batch):
        word_ids = encoded_inputs.word_ids(batch_index=i)
        if is_layering:
            raw_tags_outer = item['outer_tags']
            raw_tags_inner = item['inner_tags']
        else:
            etype = 'PER' # 简化演示
            raw_tags_outer = item.get(f'{etype}_tags', ['O'] * len(item['tokens']))
            raw_tags_inner = raw_tags_outer # 简化
            
        previous_word_idx = None
        label_ids_outer = []
        label_ids_inner = []
        
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)
            elif word_idx < len(raw_tags_outer):
                label_ids_outer.append(tag_map.get(raw_tags_outer[word_idx], -100))
                # Cascading 映射简化
                tag_map_single = {'O': 0, f'B-{etype}': 1, f'I-{etype}': 2}
                inner_map = tag_map if is_layering else tag_map_single
                label_ids_inner.append(inner_map.get(raw_tags_inner[word_idx], -100))
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

# --- 2. Span-Based Data Collator ---
def data_collator_span_based(batch, tokenizer, span_label_map, max_spans=100):
    if len(batch) > 1: raise ValueError("Span-Based batch_size=1 only for demo.")
    item = batch[0]
    tokens = item['tokens']
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt')
    
    all_spans = item['spans']
    selected_spans = all_spans[:max_spans] 
    
    start_indices = [s['span'][0] + 1 for s in selected_spans] 
    end_indices = [s['span'][1] for s in selected_spans]
    span_widths = [s['span'][1] - s['span'][0] for s in selected_spans]
    span_labels_ids = [span_label_map.get(s['label'], 0) for s in selected_spans]
    
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(start_indices), torch.tensor(end_indices), torch.tensor(span_widths), torch.tensor(span_labels_ids)

# --- 3. Generative Data Collator (ReasoningIE) ---
def data_collator_generative(batch, tokenizer):
    inputs = [item['input_text'] for item in batch]
    targets = [item['target_text'] for item in batch]
    
    # 输入 Tokenization
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    
    # 目标 Tokenization (作为 labels)
    # 使用 tokenizer 作为 target_tokenizer (BERT tokenizer 通用)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_GEN_LEN, padding='max_length', truncation=True, return_tensors='pt')
    
    # 将 Pad Token 的 Label 设为 -100，忽略 Loss
    labels_ids = labels['input_ids']
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100
    
    return model_inputs['input_ids'], model_inputs['attention_mask'], labels_ids

# --- 真实数据集类 ---
class NERDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 训练与评估逻辑 ---
def train_and_evaluate(method_name, model, dataset, data_collator_fn, tag_map, num_epochs=3, learning_rate=2e-5):
    print(f"\n--- 启动 {method_name} 方法训练 ---")
    print(f"数据集大小: {len(dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Collator 选择
    if "Layering" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=True)
        batch_size = 4
    elif "Cascading" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=False)
        batch_size = 4
    elif "Span-Based" in method_name:
        collator = lambda batch: data_collator_span_based(batch, tokenizer, tag_map)
        batch_size = 1 
    elif "ReasoningIE" in method_name:
        collator = lambda batch: data_collator_generative(batch, tokenizer)
        batch_size = 4 # 生成式可以支持 batch
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(dataloader):
            batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            optimizer.zero_grad()
            
            if "Layering" in method_name:
                input_ids, attention_mask, labels_outer, labels_inner = batch
                loss, _, _ = model(input_ids, attention_mask, labels_outer, labels_inner)
            elif "Cascading" in method_name:
                input_ids, attention_mask, labels = batch 
                loss, _ = model(input_ids, attention_mask, labels)
            elif "Span-Based" in method_name:
                input_ids, attention_mask, start, end, widths, spans = batch
                loss, _ = model(input_ids, attention_mask, start, end, widths, spans)
            elif "ReasoningIE" in method_name:
                input_ids, attention_mask, labels = batch
                loss, _ = model(input_ids, attention_mask, labels=labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                 print(f"  Batch {i+1}/{len(dataloader)} Loss: {total_loss / (i+1):.4f}")

    print(f"训练完成：{method_name}。")

# --- 主实验函数 ---
def run_experiment_pipeline(data_files):
    train_file = data_files[0]
    if not os.path.exists(train_file):
        print(f"错误: 找不到文件 {train_file}")
        return

    print(f"正在加载数据: {train_file} ...")
    raw_data = load_data(train_file)
    print("正在预处理数据...")
    processed_data = preprocess_for_all_methods(raw_data)
    
    # 1. Layering
    layering_model = LayeringNERModel()
    train_and_evaluate("Layering Method", layering_model, NERDataset(processed_data['layering']), None, TAG_TO_ID)

    # 2. Cascading (PER)
    cascading_model = SingleTypeNERModel('PER')
    train_and_evaluate("Cascading Method (PER)", cascading_model, NERDataset(processed_data['cascading']), None, TAG_TO_ID)

    # 3. Span-Based
    span_model = SpanBasedNERModel()
    train_and_evaluate("Span-Based Method", span_model, NERDataset(processed_data['span_based']), None, SPAN_LABEL_TO_ID)

    # 4. ReasoningIE (Generative)
    print("\n------------------------------------------------------------")
    print(f"开始运行 ReasoningIE Method (Generative) 训练")
    print("------------------------------------------------------------")
    reasoning_model = ReasoningIENERModel()
    train_and_evaluate("ReasoningIE Method", reasoning_model, NERDataset(processed_data['reasoning_ie']), None, None)

if __name__ == '__main__':  
    data_path = os.path.join("data", "train.jsonlines")
    run_experiment_pipeline([data_path])