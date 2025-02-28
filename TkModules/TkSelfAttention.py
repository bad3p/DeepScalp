import torch

# --------------------------------------------------------------------------------------------------------------
# Self-attention module

class TkSelfAttention(torch.nn.Module):

    def __init__(self, _input_size : int, _history_size : int, _attention_size : int, _normalization : bool):
        super(TkSelfAttention, self).__init__()

        self._input_size = _input_size
        self._history_size = _history_size
        self._attention_size = _attention_size
        self._normalization = _normalization

        attention_input_size = self._input_size * self._history_size
        self._attention_query = torch.nn.Linear(attention_input_size, self._attention_size)
        self._attention_key = torch.nn.Linear(attention_input_size, self._attention_size)
        self._attention_value = torch.nn.Linear(attention_input_size, self._attention_size)
        self._attention_softmax = torch.nn.Softmax(dim=1)        

        self.initWeights()

    def initWeights(self) -> None:

        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(p.data)
                    #torch.nn.init.normal_(p.data, 0.0, 0.2)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(p.data)
                    #torch.nn.init.normal_(p.data, 0.0, 0.2)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.1)
                torch.nn.init.constant_(m.bias, 0)
            

    def forward(self, input):

        batch_size = input.shape[0]

        attention_input_size = self._input_size * self._history_size
        attention_input = input.reshape( (batch_size, attention_input_size))
        queries = self._attention_query( attention_input )
        keys = self._attention_key( attention_input )
        values = self._attention_value( attention_input )
        queries = queries.reshape( (batch_size, self._attention_size, 1) )
        keys = keys.reshape( (batch_size, self._attention_size, 1) )
        values = values.reshape( (batch_size, self._attention_size, 1) )
        scores = torch.bmm(queries, keys.transpose(1, 2)) / ((self._attention_size ** 0.5) if self._normalization else 1.0)
        scores = scores.reshape( (batch_size, self._attention_size * self._attention_size ) )
        attention = self._attention_softmax(scores)
        attention = attention.reshape( (batch_size, self._attention_size, self._attention_size) )
        attention_output = torch.bmm(attention, values)
        attention_output = attention_output.reshape( (batch_size, self._attention_size) )

        return attention_output

    def hiddenNeuronCount(self):
        return self._attention_size * 3 + self._attention_size * self._attention_size