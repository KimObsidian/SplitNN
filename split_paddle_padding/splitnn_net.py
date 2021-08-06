import torch,random
from torch.distributions.laplace import Laplace
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 30
epsilon = 4.0
eps = epsilon / epoch
mean = torch.zeros(32, 128)

class SplitNN(torch.nn.Module):
    def __init__(self, models, optimizers, data_owner):
        self.data_owners = data_owner
        self.optimizers = optimizers
        self.models = models
        self.activation=[]
        self.count=[0,0,0,0]
        self.print=False
        super().__init__()


    def forward(self, data_pointer):
        client_output = {}
        remote_outputs = []
        i = 0
        for owner in self.data_owners:
            if i == 0:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                #calculate one normalization
                one_norm = torch.norm(client_output[owner], p = 1, dim = 1)
                #calculate sensitivity
                sensitivity = (torch.amax(one_norm, dim = 0) - torch.amin(one_norm, dim = 0)).item()
                #calculate noise multiplier
                noise_mul = sensitivity / eps
                scale = torch.full((32, 128), noise_mul)
                lap = Laplace(mean, scale)
                #sample noise
                noise = lap.sample()
                #add noise
                client_output[owner] = torch.add(client_output[owner], noise)
                client_output[owner].to(device)
                #print(client_output[owner].requires_grad)
                value_list = [0, 1]
                probability = [0.05, 0.95]
                random_response = []
                for i in range(128):
                    temp = self.number_of_certain_probability(value_list, probability)
                    random_response.append(temp)
                random_response = torch.tensor(random_response)
                client_output[owner] = torch.mul(client_output[owner], random_response)
                remote_outputs.append(client_output[owner].requires_grad_().to(device))
                i += 1
            else:
                self.models[owner].to(device)
                client_output[owner] = self.models[owner](data_pointer[owner].reshape([-1, 160]).to(device))
                client_output[owner].to(device)
                remote_outputs.append(client_output[owner].requires_grad_())
        server_input = torch.min(remote_outputs[0],remote_outputs[1])
        #server_input = torch.cat(remote_outputs,1)
        #server_input = torch.max(remote_outputs[0],remote_outputs[1])
        #server_input = torch.add(remote_outputs[0],remote_outputs[1])
        #server_input = torch.add(remote_outputs[0]/2, remote_outputs[1]/2)
        self.models["server"].to(device)
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

    def number_of_certain_probability(self,sequence, probability):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item
