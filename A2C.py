import torch
import torch.nn.functional as F

class A2C(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[2], out_channels=32, kernel_size=4, stride=2)
        dim1 = ((state_size[0] - 4)//2 + 1, (state_size[1] - 4)//2 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        dim2 = ((dim1[0] - 3)//2 + 1, (dim1[1] - 3)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        dim3 = ((dim2[0] - 2)//1 + 1, (dim2[1] - 2)//1 + 1)

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], 128)

        self.critic_linear = torch.nn.Linear(128, 1)
        self.actor_linear = torch.nn.Linear(128, action_size)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        
        
        policy = F.softmax(self.actor_linear(x), dim=1)
        value = self.critic_linear(x)
        return policy, value