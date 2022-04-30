import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, output_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, output_classes)

    def forward(self, x):
        logit = self.linear(x)
        out = torch.softmax(logit, dim=1)
        return out
