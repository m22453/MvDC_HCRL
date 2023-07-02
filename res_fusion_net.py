
import torch
from torch import nn

class ResFusionWithGLU(nn.Module):
    def __init__(self, in_dims) -> None:
        super().__init__()

        # multi-view attention fusion layer
        self.mvf_layer = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.ReLU(),
            nn.Linear(in_dims, in_dims)
        ) # W_f 矩阵
        self.hidden_sigmoid = nn.Linear(in_dims, 1)

        if in_dims % 2 == 1:
            raise ValueError('in_dims must be an even')
        
        DOUBLE_DIM = int(in_dims * 4) # BBC=6 
        HALF_DIM = int(in_dims * 0.5)
        
        self.magnify_block = nn.Sequential(
            nn.Linear(in_dims, DOUBLE_DIM),
            nn.BatchNorm1d(DOUBLE_DIM),
            nn.ReLU(),
            nn.Linear(DOUBLE_DIM, in_dims),
            nn.BatchNorm1d(in_dims)
        )

        self.magnify_activate = nn.ReLU()


        self.shrink_block = nn.Sequential(
            nn.Linear(in_dims, HALF_DIM),
            nn.BatchNorm1d(HALF_DIM),
            nn.ReLU(),
            nn.Linear(HALF_DIM, in_dims),
            nn.BatchNorm1d(in_dims)

        )
        
        self.shrink_activate = nn.ReLU()

        self.fusion_glu = nn.GLU()

    def forward(self, x, xs=None):
        """
        x: the view-common representation by view adding
        """            

        identify = x.clone()

        if not xs == None:
            xx_ = torch.zeros_like(identify)
            for xx in xs:
                xx = self.mvf_layer(xx) # for sharing
                # xx_prob = nn.functional.sigmoid(self.hidden_sigmoid(xx)) # for gate
                xx_ = xx_ + xx * 1.0 / len(xs) 

            gate_prob = nn.functional.sigmoid(self.hidden_sigmoid(xx_))
            identify = identify + xx_ * gate_prob
                

        # for magnify op
        x = self.magnify_block(x)

        # skip connection
        x += identify

        # activate
        x = self.magnify_activate(x)

        identify_ = x.clone()

        # for shrink op
        x = self.shrink_block(x)

        # skip connection
        x += identify_

        # activate
        x = self.shrink_activate(x)

        # GLU fusion
        f_x = self.fusion_glu(torch.cat((identify, x), dim=-1)) + identify
        f_x = f_x + self.fusion_glu(torch.cat((identify, identify_), dim=-1)) + identify


        return (identify_,  x, f_x) # original x, output of res_block, view fusion of GLU


if __name__ == "__main__":

    pass
    x = torch.rand(2, 10)

    net = ResFusionWithGLU(x.shape[-1])

    for it in net(x):
        print(it.size())

