import torch

# Original Extreme Learning Machine implementation
class ELM:
    def __init__(self, inp, hid, out, no_init = True):
        """
        inp: int, size of the input vector
        hid: int, number of the hidden units
        out: int, number of output classes
        device: str, gpu or cpu
        returns: None
        """
        # Could be non-orthogonal too
        if no_init:
            self.w = torch.empty((inp, hid))  
            print("ale")
        else:
            self.w = torch.nn.init.orthogonal_(torch.empty(inp, hid))
            print("molto male")

        self.b = torch.rand(1, hid)
        self.beta = torch.rand(hid, out)
    
    def embed(self, x):
        """
        x: tensor, the input data
        returns: tensor, output scores
        """
    
        x = x.view(x.shape[0], -1)
        x = torch.relu((x @ self.w) + self.b)
        return x

    def to(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.beta = self.beta.to(device)
        return self

    def parameters(self):
        return []
    
    def state_dict(self):
        dic = {}
        dic["w"]  = self.w
        dic["b"]  = self.b
        dic["beta"]  = self.beta
        return dic
    
    def load_state_dict(self, dict, strict=False):
        self.w = dict['w']
        self.b = dict['b']
        self.beta = dict['beta']

    def train(self):
        #print("train mode: ACTIVATED")
        pass
    
    def forward(self, x):
        """
        x: tensor, the input data
        returns: tensor, output scores
        """
        with torch.no_grad():
            temp = torch.relu((x @ self.w) + self.b)    
        return temp

    def fit(self, x, y):
        """
        x: tensor, the training data
        returns: None
        """
        # y must be one hot encoded
        self.beta = torch.pinverse(self.forward(x)) @ y

    def predict(self, x):
        """
        x: tensor, the test data
        returns: None
        """
        return self.forward(x) @ self.beta

