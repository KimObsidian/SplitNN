import numpy as np


class pytorch_pruning(object):
    def __init__(self, parameters, pruning_settings=dict(), log_folder=None):
        self.temp_hessian = []
        self.parameters = list()
        for parameter in parameters:
            paramter_value=parameter["parameter"]
            self.parameters.append(paramter_value)
            self.prune_network_criteria = list()
            self.prune_network_accomulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}
            self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accomulate.keys():
                self.prune_network_accomulate[key].append(list())

            self.pruning_gates.append(np.ones(len(self.parameters[layer]), ))
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)




if __name__=="__main__":
    print([1,2]*[1,2])



