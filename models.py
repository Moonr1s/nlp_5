import torch
import torch.nn as nn
from transformers import BertModel, EncoderDecoderModel
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
        # 注意：这里仅演示 batch_size=1 的逻辑，实际需要更复杂的 batch indexing
        if sequence_output.size(0) > 1:
            raise NotImplementedError("Span-Based 模型的 Data Collator 演示仅支持 batch_size=1。")

        h_start = sequence_output[0, start_indices]
        h_end = sequence_output[0, end_indices]

        # b) 计算 Span Average Pooling
        span_reps = []
        for i, (s, e) in enumerate(zip(start_indices, end_indices)):
            # [Fix] 修改切片范围为 s : e + 1，否则单字实体会变成空切片 (NaN)，且多字实体会丢失最后一个字
            h_span_avg = torch.mean(sequence_output[0, s : e + 1], dim=0)
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

# --- 4. ReasoningIE-Style Model (Generative / Seq2Seq) ---
class ReasoningIENERModel(nn.Module):
    """
    模型 4: ReasoningIE Method (Generative / Seq2Seq)
    参考 HuiResearch/ReasoningIE，使用生成式框架。
    这里使用 EncoderDecoderModel (BERT-to-BERT) 来模拟生成过程。
    输入: 原始句子
    输出: 实体描述字符串 (例如: "人名: 张三; 地名: 北京")
    """
    def __init__(self):
        super().__init__()
        # 使用 BERT 初始化 Encoder 和 Decoder (BERT2BERT)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME)
        
        # 配置生成参数 (bert-base-chinese: [CLS]=101, [SEP]=102, [PAD]=0)
        self.model.config.decoder_start_token_id = 101 # [CLS]
        self.model.config.eos_token_id = 102       # [SEP]
        self.model.config.pad_token_id = 0         # [PAD]
        
        # 确保词汇表大小匹配
        self.model.config.vocab_size = self.model.config.encoder.vocab_size

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播函数 (Fix: 之前版本缺失此方法导致报错)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits