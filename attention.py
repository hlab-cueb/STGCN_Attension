import torch
import torch.nn as nn

#### 定义的multi-head的attention的模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, input, context=None):
        batch_size = input.size(0)
        seq_len = input.size(1)
        # Linear transformations
        if context is not None:
            # 代表的是 cross_attention
            query = self.query_linear(context)
        else:
            query = self.query_linear(input)
        
        key = self.key_linear(input)
        value = self.value_linear(input)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        key = key.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        value = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        attention_weights = nn.Softmax(dim=-1)(scores)
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, d_k]

        # Concatenate and reshape
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Linear transformation for final output
        attention_output = self.output_linear(attention_output)

        return attention_output