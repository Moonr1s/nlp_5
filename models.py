import torch
import torch.nn as nn
from transformers import BertModel
from constants import EMBED_DIM, ENTITY_TYPES, NUM_SEQ_LABELS, NUM_SPAN_LABELS, MODEL_NAME, MAX_SPAN_LEN

# --- 1. Layering Method Model (外层/内层 Sequential Tagging) ---
class LayeringNERModel(nn.Module):
    """
    模型 1: Layering Method (Sequential Tagging - BIO)
    """
    def __init__(self):
        super().__init__()
        # 使用 Hugging Face 的 BERT 作为 Encoder
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        
        # 可选：增加 BiLSTM 层来捕获更复杂的局部依赖
        self.bilstm = nn.LSTM(EMBED_DIM, EMBED_DIM // 2, 
                              bidirectional=True, batch_first=True)
        
        # 序列标注头 - 外层
        self.tagger_outer = nn.Linear(EMBED_DIM, NUM_SEQ_LABELS)
        # 序列标注头 - 内层
        self.tagger_inner = nn.Linear(EMBED_DIM, NUM_SEQ_LABELS)
        
        # 损失函数 (忽略 Padding Token 的损失，假设 Padding ID 为 -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels_outer=None, labels_inner=None):
        # 1. BERT Encoding
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] # (batch_size, seq_len, embed_dim)
        
        # 2. (Optional) BiLSTM layer
        # sequence_output, _ = self.bilstm(sequence_output) 
        
        # 3. 预测 Logits
        logits_outer = self.tagger_outer(sequence_output)
        logits_inner = self.tagger_inner(sequence_output)
        
        if labels_outer is not None and labels_inner is not None:
            # 训练模式：计算总损失
            loss_outer = self.criterion(logits_outer.view(-1, NUM_SEQ_LABELS), labels_outer.view(-1))
            loss_inner = self.criterion(logits_inner.view(-1, NUM_SEQ_LABELS), labels_inner.view(-1))
            total_loss = loss_outer + loss_inner
            return total_loss, logits_outer, logits_inner
            
        return logits_outer, logits_inner


# --- 2. Cascading Method Model (每个实体类型一个独立的模型) ---
# Cascading模型与 Layering/单个序列标注模型结构相似，只是针对一个实体类型。
class SingleTypeNERModel(nn.Module):
    def __init__(self, entity_type):
        super().__init__()
        self.entity_type = entity_type
        # 共享 BERT Encoder
        self.bert = BertModel.from_pretrained(MODEL_NAME) 
        # 独立的 Tagging Head (例如，只识别 B-PER, I-PER, O，所以标签数是 3)
        self.tagging_head = nn.Linear(EMBED_DIM, 3) 
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.tagging_head(sequence_output)
        
        if labels is not None:
            loss = self.criterion(logits.view(-1, 3), labels.view(-1))
            return loss, logits
            
        return logits


# --- 3. Enumeration/Span-Based Method Model (跨度枚举) ---
class SpanBasedNERModel(nn.Module):
    """
    模型 3: Span-Based Method (枚举所有可能的跨度并分类)
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.span_width_embedding = nn.Embedding(MAX_SPAN_LEN + 1, 50) # 跨度宽度Embedding
        
        # Span Representation: [h_start; h_end; h_average; span_width_emb]
        # EMBED_DIM * 3 (start, end, average) + 50 (width emb)
        classifier_input_dim = EMBED_DIM * 3 + 50
        
        self.span_classifier = nn.Linear(classifier_input_dim, NUM_SPAN_LABELS)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels=None):
        # 1. BERT Encoding
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] # (batch_size, seq_len, embed_dim)

        # 2. 构造 Span Representation
        # 批次维度展平：将 (Batch * Span_Count) 作为新的批次大小
        
        # a) 获取 Start/End 向量 (使用 torch.gather/index_select 或更简单的 view/indexing)
        # 这里需要复杂的批次索引操作，我们使用简单的索引来实现逻辑演示:
        
        # 重新组织 sequence_output 以便索引 (简化的单句处理逻辑，实际需要批量处理)
        if sequence_output.size(0) > 1:
            raise NotImplementedError("Span-Based 模型的批处理索引实现复杂，请参考官方实现。")

        h_start = sequence_output[0, start_indices]
        h_end = sequence_output[0, end_indices]

        # b) 计算 Span Average/Max Pooling
        # 由于批量大小为 1，可以简化计算
        span_reps = []
        for i, (s, e) in enumerate(zip(start_indices, end_indices)):
            # Average Pooling of tokens within the span [s:e]
            h_span_avg = torch.mean(sequence_output[0, s:e], dim=0)
            span_reps.append(h_span_avg)
        h_span_avg = torch.stack(span_reps)
        
        # c) 获取 Span Width Embedding
        width_emb = self.span_width_embedding(span_widths)
        
        # d) 拼接 Span Representation
        span_representation = torch.cat([h_start, h_end, h_span_avg, width_emb], dim=-1)

        # 3. 分类
        logits = self.span_classifier(span_representation) # (Num_Spans, Num_Labels)
        
        if span_labels is not None:
            # 训练模式：计算损失
            loss = self.criterion(logits, span_labels)
            return loss, logits
            
        return logits

# --- 4. ReasoningIE-Style Model (SOTA Span-Based with Enhancement) ---
class ReasoningIENERModel(SpanBasedNERModel):
    """
    模型 4: ReasoningIE-Style (SOTA Span-Based)
    继承 Span-Based 结构，并增加一个 Transformer Encoder/GNN 来进行推理。
    """
    def __init__(self):
        super().__init__()
        # 增加一个 Transformer Encoder Layer (模拟 Reasoning Layer/GNN)
        transformer_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, batch_first=True)
        self.reasoning_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        # 调整分类器输入维度: 引入 Reasoning Layer 的输出
        # [h_start; h_end; h_average; h_reasoned_start; width_emb] (维度假设)
        classifier_input_dim = EMBED_DIM * 4 + 50
        self.span_classifier = nn.Linear(classifier_input_dim, NUM_SPAN_LABELS)

    def forward(self, input_ids, attention_mask, start_indices, end_indices, span_widths, span_labels=None):
        # 1. BERT Encoding
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        # 2. Reasoning Layer
        # 对 BERT 输出进行推理，增强上下文信息
        reasoned_output = self.reasoning_encoder(sequence_output, src_key_padding_mask=~attention_mask.bool())

        # 3. Span Representation Generation (使用更丰富的表示)
        
        # 仅为演示：使用 reasoned_output 的 start/end 向量代替原始 BERT 输出
        # 简化索引逻辑 (假设 batch_size=1)
        if reasoned_output.size(0) > 1:
            raise NotImplementedError("复杂的批处理索引已省略。")
            
        h_start = reasoned_output[0, start_indices]
        h_end = reasoned_output[0, end_indices]

        # 重新计算 span average using reasoned_output
        span_reps = []
        for i, (s, e) in enumerate(zip(start_indices, end_indices)):
            h_span_avg = torch.mean(reasoned_output[0, s:e], dim=0)
            span_reps.append(h_span_avg)
        h_span_avg = torch.stack(span_reps)
        
        width_emb = self.span_width_embedding(span_widths)

        # 拼接：[reasoned_h_start; reasoned_h_end; reasoned_h_avg; width_emb]
        span_representation = torch.cat([h_start, h_end, h_span_avg, width_emb], dim=-1)

        # 4. 分类
        logits = self.span_classifier(span_representation)
        
        if span_labels is not None:
            loss = self.criterion(logits, span_labels)
            return loss, logits
            
        return logits