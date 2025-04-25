import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, login, PyTorchModelHubMixin
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
token = os.getenv('HF_READ')

# Login with token
login(token=token)

# Download model components
repo_id = "aiguy68/neurosam-model"
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)

# Download LDA components
vectorizer_path = hf_hub_download(repo_id=repo_id, filename="count_vectorizer.pkl", token=token)
lda_model_path = hf_hub_download(repo_id=repo_id, filename="lda_model.pkl", token=token)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(lda_model_path, 'rb') as f:
    lda_model = pickle.load(f)

class MultiGranularityTopicModule(nn.Module):
    def __init__(self, lda_topics=15, lda_topics_25=25, hidden_size=768, dropout_rate=0.1):
        super().__init__()
        self.topic_encoder = nn.Sequential(
            nn.Linear(lda_topics, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2)
        )
        
        self.topic_encoder_25 = nn.Sequential(
            nn.Linear(lda_topics_25, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2)
        )
        
        self.projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, topic_dist, topic_dist_25=None):
        topic_dist = torch.clamp(topic_dist, min=0.05)
        topic_dist = topic_dist / torch.sum(topic_dist, dim=1, keepdim=True)
        uniform_dist = torch.ones_like(topic_dist) / topic_dist.size(1)
        topic_dist = 0.7 * topic_dist + 0.3 * uniform_dist
        features_15 = self.topic_encoder(topic_dist)
        
        if topic_dist_25 is not None:
            topic_dist_25 = torch.clamp(topic_dist_25, min=0.05)
            topic_dist_25 = topic_dist_25 / torch.sum(topic_dist_25, dim=1, keepdim=True)
            uniform_dist_25 = torch.ones_like(topic_dist_25) / topic_dist_25.size(1)
            topic_dist_25 = 0.7 * topic_dist_25 + 0.3 * uniform_dist_25
            features_25 = self.topic_encoder_25(topic_dist_25)
            
            combined_features = torch.cat([features_15, features_25], dim=1)
            return self.projection(combined_features)
        else:
            features_placeholder = torch.zeros_like(features_15)
            combined_features = torch.cat([features_15, features_placeholder], dim=1)
            return self.projection(combined_features)

class UFENModule(nn.Module):
    def __init__(self, hidden_size=768, metadata_dim=15):
        super().__init__()
        self.bert = AutoModel.from_pretrained("albert-base-v2")
        self.projection = nn.Linear(self.bert.config.hidden_size, hidden_size)
        
    def forward(self, input_ids, attention_mask, metadata=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state
        text_features = self.projection(text_features)
        mask = attention_mask.unsqueeze(-1).float()
        text_features = text_features * mask
        return text_features

class ECOSAMModule(nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.context_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.sentiment_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)
        self.sentiment_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.sent_dropout = nn.Dropout(dropout_rate)
        self.sent_gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.context_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, sequence_features):
        lstm_out, _ = self.bilstm(sequence_features)
        context_attn, _ = self.context_attention(lstm_out, lstm_out, lstm_out)
        context_attn = self.context_dropout(context_attn)
        context_enhanced = self.attention_norm(context_attn + lstm_out)
        batch_size = sequence_features.size(0)
        sent_query = self.sentiment_query.expand(batch_size, 1, -1)
        sent_context, _ = self.sentiment_attention(sent_query, context_enhanced, context_enhanced)
        sent_context = self.sent_dropout(sent_context).squeeze(1)
        sent_gates = self.sent_gate(context_enhanced)
        sent_gates = sent_gates / (sent_gates.sum(dim=1, keepdim=True) + 1e-9)
        context_vector = (context_enhanced * sent_gates).sum(dim=1)
        return context_vector

class BidirectionalFusion(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.text_to_topic = nn.Linear(hidden_size, hidden_size)
        self.topic_to_text = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, text_representation, topic_features):
        text_for_topics = self.text_to_topic(text_representation)
        topics_for_text = self.topic_to_text(topic_features)
        
        gate = torch.sigmoid(self.gate)
        enhanced_text = text_representation + gate * topics_for_text
        enhanced_topic = topic_features + (1 - gate) * text_for_topics
        
        return enhanced_text, enhanced_topic

class StochasticDepth(nn.Module):
    def __init__(self, p=0.1, mode='batch'):
        super().__init__()
        self.p = p
        self.mode = mode
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        survival_rate = 1.0 - self.p
        size = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.bernoulli(torch.ones(size, device=x.device) * survival_rate)
        return x * mask / survival_rate

class HybridSentimentModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=3, lda_topics=15, lda_topics_25=25, 
                 hidden_size=768, dropout_rate=0.1, bert_model_name="albert-base-v2", **kwargs):
        super().__init__()
        self.config = {
            "num_classes": num_classes,
            "lda_topics": lda_topics,
            "lda_topics_25": lda_topics_25,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "bert_model_name": bert_model_name
        }
        self.config.update(kwargs)
        
        self.ufen = UFENModule(hidden_size=hidden_size, metadata_dim=lda_topics)
        self.topic_module = MultiGranularityTopicModule(
            lda_topics, lda_topics_25, hidden_size, dropout_rate=dropout_rate)
        self.ecosam = ECOSAMModule(hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.bidir_fusion = BidirectionalFusion(hidden_size=hidden_size)
        self.drop_path = StochasticDepth(p=dropout_rate, mode='batch')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            StochasticDepth(p=dropout_rate, mode='batch'),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, topic_dist, topic_dist_25=None):
        topic_features = self.topic_module(topic_dist, topic_dist_25)
        sequence_features = self.ufen(input_ids, attention_mask)
        context_vector = self.ecosam(sequence_features)
        text_representation = context_vector
        enhanced_text, enhanced_topic = self.bidir_fusion(text_representation, topic_features)
        fused_features = enhanced_text + enhanced_topic
        logits = self.classifier(fused_features)
        return logits

def adapt_topic_dist(dist, target_size):
    batch_size = dist.shape[0]
    result = torch.zeros((batch_size, target_size), dtype=torch.float)
    topics_to_copy = min(dist.shape[1], target_size)
    result[:, :topics_to_copy] = torch.tensor(dist[:, :topics_to_copy], dtype=torch.float)
    result = result / torch.sum(result, dim=1, keepdim=True)
    return result
