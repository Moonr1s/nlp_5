import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.optim import AdamW
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

# 导入所有常量和模型
from constants import TAG_TO_ID, SPAN_LABEL_TO_ID, ENTITY_TYPES, MODEL_NAME, MAX_GEN_LEN
from models import LayeringNERModel, SingleTypeNERModel, SpanBasedNERModel, ReasoningIENERModel

# 导入数据处理模块
from data_processor import load_data, preprocess_for_all_methods

# 反转 ID 映射，用于解码预测结果
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}
ID_TO_SPAN_LABEL = {v: k for k, v in SPAN_LABEL_TO_ID.items()}

# --- 辅助函数：F1 计算 ---
def calculate_f1(pred_set, gold_set, prefix=""):
    # pred_set 和 gold_set 都是集合 {(type, start, end), ...} 或 {(type, text), ...}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"[{prefix}] P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    return f1

def extract_entities_bio(tag_ids):
    """从 BIO 标签序列中提取实体 (type, start, end)"""
    entities = set()
    tags = [ID_TO_TAG.get(i, "O") for i in tag_ids]
    
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f"I-{etype}":
                i += 1
            end = i
            entities.add((etype, start, end))
        else:
            i += 1
    return entities

def extract_entities_span(start_indices, end_indices, label_ids):
    """从 Span 模型输出提取实体"""
    entities = set()
    # start_indices, end_indices 是 tensor
    for s, e, l in zip(start_indices, end_indices, label_ids):
        l = l.item()
        if l == 0: continue # O label
        etype = ID_TO_SPAN_LABEL.get(l)
        if etype:
            entities.add((etype, s.item(), e.item()))
    return entities

def parse_generative_output(text):
    """解析 'PER: 张三; LOC: 北京' 格式"""
    entities = set()
    if not text or text == "无实体": return entities
    parts = text.split(";")
    for part in parts:
        if ":" in part:
            try:
                etype, content = part.split(":", 1)
                entities.add((etype.strip(), content.strip()))
            except:
                continue
    return entities

# --- 核心评估函数 ---
def evaluate_model(method_name, model, dataloader, device, tokenizer):
    print(f"\n正在评估 {method_name} ...")
    model.eval()
    
    all_preds = [] # 每个样本一个集合
    all_golds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            
            if "Layering" in method_name:
                input_ids, attention_mask, labels_outer, labels_inner = batch
                # 模型返回 logits
                _, logits_outer, logits_inner = model(input_ids, attention_mask, labels_outer, labels_inner)
                
                # 解码 (Batch size 循环)
                pred_ids_inner = torch.argmax(logits_inner, dim=-1).cpu().numpy()
                lbl_ids_inner = labels_inner.cpu().numpy()
                
                for i in range(len(input_ids)):
                    # 简单起见，Layering 演示只评估 Inner Tags (细粒度实体)
                    # 过滤掉 -100 (Padding)
                    valid_len = (lbl_ids_inner[i] != -100).sum()
                    p_tags = pred_ids_inner[i][:valid_len]
                    g_tags = lbl_ids_inner[i][:valid_len]
                    
                    all_preds.append(extract_entities_bio(p_tags))
                    all_golds.append(extract_entities_bio(g_tags))

            elif "Cascading" in method_name:
                input_ids, attention_mask, labels = batch
                loss, logits = model(input_ids, attention_mask, labels) # Cascading forward 返回 loss, logits
                
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                lbl_ids = labels.cpu().numpy()
                
                for i in range(len(input_ids)):
                    valid_len = (lbl_ids[i] != -100).sum()
                    p_tags = pred_ids[i][:valid_len]
                    g_tags = lbl_ids[i][:valid_len]
                    # Cascading 模型只预测单一类型 (如 PER)，这里直接提取
                    all_preds.append(extract_entities_bio(p_tags))
                    all_golds.append(extract_entities_bio(g_tags))
            
            elif "Span-Based" in method_name:
                input_ids, attention_mask, start, end, widths, span_labels = batch
                logits = model(input_ids, attention_mask, start, end, widths) # 评估时 Forward 不传 labels
                
                pred_labels = torch.argmax(logits, dim=-1)
                # 这是一个 Batch 的所有 spans，实际上 span-based dataloader 这里 batch=1
                # 我们直接把这一整句的所有预测 Span 收集起来
                all_preds.append(extract_entities_span(start, end, pred_labels))
                all_golds.append(extract_entities_span(start, end, span_labels))

            elif "ReasoningIE" in method_name:
                input_ids, attention_mask, labels = batch
                # 生成式解码
                generated_ids = model.model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    max_length=MAX_GEN_LEN
                )
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # 解码 Labels (用于对比)
                labels[labels == -100] = tokenizer.pad_token_id # 还原 -100
                decoded_golds = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                for p_text, g_text in zip(decoded_preds, decoded_golds):
                    all_preds.append(parse_generative_output(p_text))
                    all_golds.append(parse_generative_output(g_text))

    # 汇总计算
    total_pred = set()
    total_gold = set()
    
    # 简单的全局 Set F1 (Micro-F1)
    # 注意：为了区分不同句子中的相同实体，我们需要给实体加上句子索引
    for idx, (p_set, g_set) in enumerate(zip(all_preds, all_golds)):
        for item in p_set:
            total_pred.add((idx, item)) # (sent_idx, (type, start, end) OR (type, text))
        for item in g_set:
            total_gold.add((idx, item))
            
    calculate_f1(total_pred, total_gold, prefix=method_name)
    model.train() # 切回训练模式

# --- Data Collators (保持之前修复的版本) ---
def data_collator_sequence(batch, tokenizer, tag_map, is_layering=False):
    tokens = [item['tokens'] for item in batch]
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
    
    batch_labels_outer = []
    batch_labels_inner = []
    
    for i, item in enumerate(batch):
        word_ids = encoded_inputs.word_ids(batch_index=i)
        if is_layering:
            raw_tags_outer = item['outer_tags']
            raw_tags_inner = item['inner_tags']
            etype = None 
        else:
            etype = 'PER' 
            raw_tags_outer = item.get(f'{etype}_tags', ['O'] * len(item['tokens']))
            raw_tags_inner = raw_tags_outer 
            
        previous_word_idx = None
        label_ids_outer = []
        label_ids_inner = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids_outer.append(-100)
                label_ids_inner.append(-100)
            elif word_idx < len(raw_tags_outer):
                label_ids_outer.append(tag_map.get(raw_tags_outer[word_idx], -100))
                if is_layering:
                    inner_val = tag_map.get(raw_tags_inner[word_idx], -100)
                else:
                    tag_map_single = {'O': 0, f'B-{etype}': 1, f'I-{etype}': 2}
                    inner_val = tag_map_single.get(raw_tags_inner[word_idx], -100)
                label_ids_inner.append(inner_val)
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

def data_collator_span_based(batch, tokenizer, span_label_map, max_spans=100):
    if len(batch) > 1: raise ValueError("Span-Based batch_size=1 only for demo.")
    item = batch[0]
    tokens = item['tokens']
    encoded_inputs = tokenizer(tokens, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
    all_spans = item['spans']
    selected_spans = all_spans[:max_spans] 
    start_indices = [s['span'][0] + 1 for s in selected_spans] 
    end_indices = [s['span'][1] for s in selected_spans]
    span_widths = [s['span'][1] - s['span'][0] for s in selected_spans]
    span_labels_ids = [span_label_map.get(s['label'], 0) for s in selected_spans]
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(start_indices), torch.tensor(end_indices), torch.tensor(span_widths), torch.tensor(span_labels_ids)

def data_collator_generative(batch, tokenizer):
    inputs = [item['input_text'] for item in batch]
    targets = [item['target_text'] for item in batch]
    model_inputs = tokenizer(inputs, max_length=256, padding=True, truncation=True, return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_GEN_LEN, padding=True, truncation=True, return_tensors='pt')
    labels_ids = labels['input_ids']
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100
    return model_inputs['input_ids'], model_inputs['attention_mask'], labels_ids

class NERDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 训练与评估逻辑 (已更新) ---
def train_and_evaluate(method_name, model, train_dataset, dev_dataset, data_collator_fn, tag_map, num_epochs=3, learning_rate=2e-5):
    print(f"\n--- 启动 {method_name} 方法训练 ---")
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(dev_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Collator 选择 (Batch Size 设为 2 防止 OOM)
    if "Layering" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=True)
        batch_size = 2 
    elif "Cascading" in method_name:
        collator = lambda batch: data_collator_sequence(batch, tokenizer, tag_map, is_layering=False)
        batch_size = 2
    elif "Span-Based" in method_name:
        collator = lambda batch: data_collator_span_based(batch, tokenizer, tag_map)
        batch_size = 1 
    elif "ReasoningIE" in method_name:
        collator = lambda batch: data_collator_generative(batch, tokenizer)
        batch_size = 2
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in progress_bar:
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
            progress_bar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

        # --- 每个 Epoch 结束后进行评估 ---
        evaluate_model(method_name, model, dev_dataloader, device, tokenizer)

    print(f"训练完成：{method_name}。")
    torch.cuda.empty_cache()

# --- 主实验函数 (加载验证集) ---
def run_experiment_pipeline(data_files):
    train_file = data_files[0]
    dev_file = os.path.join("data", "dev.jsonlines") # 假设验证集在这里

    if not os.path.exists(train_file):
        print(f"错误: 找不到文件 {train_file}")
        return
    
    if not os.path.exists(dev_file):
        print(f"警告: 找不到验证集 {dev_file}，将使用训练集的前10%作为验证。")
        raw_train = load_data(train_file)
        split_idx = int(len(raw_train) * 0.9)
        raw_dev = raw_train[split_idx:]
        raw_train = raw_train[:split_idx]
    else:
        print(f"正在加载训练数据: {train_file}")
        raw_train = load_data(train_file)
        print(f"正在加载验证数据: {dev_file}")
        raw_dev = load_data(dev_file)

    print("正在预处理数据...")
    train_processed = preprocess_for_all_methods(raw_train)
    dev_processed = preprocess_for_all_methods(raw_dev)
    
    # 1. Layering
    #layering_model = LayeringNERModel()
    #train_and_evaluate("Layering Method", layering_model, 
    #                  NERDataset(train_processed['layering']), 
    #                   NERDataset(dev_processed['layering']), 
    #                   None, TAG_TO_ID)

    # 2. Cascading (PER)
    cascading_model = SingleTypeNERModel('PER')
    train_and_evaluate("Cascading Method (PER)", cascading_model, 
                       NERDataset(train_processed['cascading']), 
                       NERDataset(dev_processed['cascading']),
                       None, TAG_TO_ID)

    # 3. Span-Based
    span_model = SpanBasedNERModel()
    train_and_evaluate("Span-Based Method", span_model, 
                       NERDataset(train_processed['span_based']), 
                       NERDataset(dev_processed['span_based']),
                       None, SPAN_LABEL_TO_ID)

    # 4. ReasoningIE (Generative)
    reasoning_model = ReasoningIENERModel()
    train_and_evaluate("ReasoningIE Method", reasoning_model, 
                       NERDataset(train_processed['reasoning_ie']), 
                       NERDataset(dev_processed['reasoning_ie']),
                       None, None)

if __name__ == '__main__':  
    data_path = os.path.join("data", "train.jsonlines")
    run_experiment_pipeline([data_path])