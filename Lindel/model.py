import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, output_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, output_classes)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.zero_()

    def forward(self, x):
        logit = self.linear(x)  # Since we are applying nn.CrossEntropyLoss, we don't need to apply softmax here
        return logit
