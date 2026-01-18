
import torch
import math

# --------------------------------------------------------------------------------------------------------------
# Self-attention module
# --------------------------------------------------------------------------------------------------------------

class TkSelfAttention(torch.nn.Module):

    def __init__(self, _inputSize : int, _attentionSize : int, _normalization : bool):
        super(TkSelfAttention, self).__init__()

        self._inputSize = _inputSize
        self._attentionSize = _attentionSize
        self._normalization = _normalization

        self._attentionQuery = torch.nn.Linear(self._inputSize, self._attentionSize, bias=False)
        self._attentionKey = torch.nn.Linear(self._inputSize, self._attentionSize, bias=False)
        self._attentionValue = torch.nn.Linear(self._inputSize, self._attentionSize, bias=False)
        self._attentionSoftmax = torch.nn.Softmax(dim=1)
        self._attentionOutput = torch.nn.Linear(self._attentionSize, self._attentionSize)

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
                    torch.nn.init.normal_(m.weight, 0, 0.1)
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
                if m.bias != None:
                    torch.nn.init.constant_(m.bias, 0)
            

    def forward(self, input):

        batchSize = input.shape[0]

        attentionInput = input.reshape( (batchSize, self._inputSize) )
        queries = self._attentionQuery( attentionInput )
        keys = self._attentionKey( attentionInput )
        values = self._attentionValue( attentionInput )
        queries = queries.reshape( (batchSize, self._attentionSize, 1) )
        keys = keys.reshape( (batchSize, self._attentionSize, 1) )
        values = values.reshape( (batchSize, self._attentionSize, 1) )
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / ((self._attentionSize ** 0.5) if self._normalization else 1.0)
        scores = scores.reshape( (batchSize, self._attentionSize * self._attentionSize ) )
        attention = self._attentionSoftmax(scores)
        attention = attention.reshape( (batchSize, self._attentionSize, self._attentionSize) )
        attentionOutput = torch.bmm(attention, values)
        attentionOutput = attentionOutput.reshape( (batchSize, self._attentionSize) )

        return self._attentionOutput( attentionOutput )

    def hiddenNeuronCount(self):
        return self._attentionSize * 3