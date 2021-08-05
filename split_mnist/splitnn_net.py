import torch


class SplitNN(torch.nn.Module):

    def __init__(self, models, optimizers, data_owner):

        self.data_owners = data_owner
        self.optimizers = optimizers
        self.models = models

        super().__init__()

    def forward(self, data_pointer):

        # individual client's output upto their respective cut layer
        client_output = {}

        # outputs that is moved to server and subjected to concatenate for server input
        remote_outputs = []

        # iterate over each client and pass thier inputs to respective model segment and move outputs to server
        i = 0
        for owner in self.data_owners:
            if i == 0:
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 14 * 28]))
                remote_outputs.append(
                    client_output[owner].requires_grad_()
                )
                i += 1
            else:
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 14 * 28]))
                remote_outputs.append(
                    client_output[owner].requires_grad_()
                )

        # concat outputs from all clients at server's location
        # server_input=torch.add(remote_outputs[0],1,remote_outputs[1])
        # server_input=server_input/2
        server_input = torch.add(remote_outputs[0], 1, remote_outputs[1])
        server_input = server_input / 2

        # pass concatenated output from server's model segment
        server_output = self.models["server"](server_input)

        return server_output

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        for loc in self.models.keys():
            for i in range(len(self.models[loc])):
                self.models[loc][i].train()

    def eval(self):
        for loc in self.models.keys():
            for i in range(len(self.models[loc])):
                self.models[loc][i].eval()
