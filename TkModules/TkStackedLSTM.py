
import torch

# --------------------------------------------------------------------------------------------------------------
# Stacked LSTM module
# --------------------------------------------------------------------------------------------------------------

class TkStackedLSTM(torch.nn.Module):

    def __init__(self, _input_size : int, _history_size : int, _specification : list):
        super(TkStackedLSTM, self).__init__()
          
        self._input_size = _input_size
        self._history_size = _history_size
        self._specification = _specification
        
        self._lstm = []

        for i in range( len(self._specification) ):
            input_size = self._input_size if i == 0 else self._specification[i-1]
            output_size = self._specification[i]
            self._lstm.append( torch.nn.LSTM( input_size, output_size, batch_first=True) )

        self._lstm = torch.nn.ModuleList( self._lstm )        

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
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
            

    def forward(self, input):

        batch_size = input.shape[0]

        lstmOutput = [None] * len(self._specification)
        mh = [None] * len(self._specification)
        mc = [None] * len(self._specification)

        for i in range(len(self._specification)):
            (mh[i], mc[i]) =  (torch.zeros(1, batch_size, self._specification[i]), torch.zeros(1, batch_size, self._specification[i]))
            mh[i] = mh[i].cuda()
            mc[i] = mc[i].cuda()
            lstmOutput[i], (mh[i], mc[i]) = self._lstm[i]( input if i == 0 else lstmOutput[i-1], (mh[i], mc[i]))

        mh[-1] = mh[-1].reshape( (batch_size, self._specification[-1]) )
        self._output = mh[-1]

        return self._output

    def hidden_neurons_count(self):
        return sum(self._specification)