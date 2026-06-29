import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1, num_layers=2):
        super(TradingLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_layer_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True 
        )
        
        self.attention = nn.Linear(hidden_layer_size * 2, 1)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        predictions = self.linear(context_vector)
        return predictions
