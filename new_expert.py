import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class Expert(nn.Module):
    def __init__(self, input_size=768, output_size=2, hidden_unit1=128, 
                 hidden_unit2=64, projected_output_size=2):
        super().__init__()    
        
        self.hidden_unit1 = hidden_unit1
        self.hidden_unit2 = hidden_unit2
        
        self.fc1 = nn.Linear(input_size, self.hidden_unit1)
        self.fc2 = nn.Linear(self.hidden_unit1, self.hidden_unit2)
        self.fc3 = nn.Linear(self.hidden_unit2, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.mapper = nn.Linear(in_features=output_size, 
                                out_features=projected_output_size)
 
        self.instance_norm1 = nn.InstanceNorm1d(num_features=self.hidden_unit1)
        self.instance_norm2 = nn.InstanceNorm1d(num_features=self.hidden_unit2)

        # initialise with xavier distribution
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        # out = F.gelu(self.instance_norm1(self.fc1(x)), approximate="tanh")
        # out = F.gelu(self.instance_norm2(self.fc2(out)), approximate="tanh")
        
        out = self.dropout(F.relu(self.instance_norm1(self.fc1(x))))
        out = self.dropout(F.relu(self.instance_norm2(self.fc2(out))))
        
        out = self.mapper(self.fc3(out))
        return out
    
class Gate(nn.Module):
    def __init__(self, output_size, input_size=768):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.instance_norm1 = nn.InstanceNorm1d(num_features=256)
        self.dropout = nn.Dropout(p=0.5)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        # out = F.gelu(self.instance_norm1(self.fc1(x)), approximate="tanh")
        out = self.dropout(F.gelu(self.instance_norm1(self.fc1(x)), approximate="tanh"))
        # out = self.dropout(F.relu(self.instance_norm1(self.fc1(x))))
        out = self.fc2(out)
        return out            

class BiasLayer(torch.nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return self.alpha * x + self.beta        

class DynamicExpert(nn.Module):
    def __init__(self, input_size=768, hidden_size=20, 
                 hidden_unit1 = 128,
                 hidden_unit2 = 64,
                 total_cls=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_cls = total_cls
        self.gate = None
        self.experts = None
        self.bias_layers = None
        self.prev_classes = []
        self.cum_classes = set()
        self.relu = nn.ReLU()

        print("input_size:", self.input_size)
        print("hidden_size:", self.hidden_size)

    def expand_expert(self, seen_cls, new_cls):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.experts:
            self.prev_classes.append(self.new_cls)
            self.gate = Gate(input_size=self.input_size, output_size=1)
            self.bias_layers = nn.ModuleList([BiasLayer()])
            
            self.experts = nn.ModuleList([Expert(input_size=self.input_size,
                                                output_size=self.new_cls,
                                                projected_output_size=self.new_cls)])                                    
            self.num_experts = len(self.experts)
        else:                        
            self.prev_classes.append(self.new_cls)
            self.gate = Gate(input_size=self.input_size, output_size=self.num_experts+1)
            self.bias_layers.append(BiasLayer())
            
            old_experts = copy.deepcopy(self.experts)
            old_experts.append(Expert(input_size=self.input_size,
                                    output_size=self.new_cls,
                                    projected_output_size=self.new_cls))
            self.experts = old_experts            
                        
            self.num_experts = len(self.experts)
            for expert_index, module in enumerate(self.experts):
                start = sum(self.prev_classes[:expert_index])
                end = start + self.prev_classes[expert_index]

                weight = module.mapper.weight
                input_size = module.mapper.in_features
                new_mapper = nn.Linear(in_features=input_size, out_features=sum(self.prev_classes), bias=False)

                with torch.no_grad():
                    all_ = {i for i in range(sum(self.prev_classes))}
                    kept_ = {i for i in range(start, end)}
                    removed_ = all_ - kept_
                    
                    upper_bound = sum(self.prev_classes[:expert_index+1])

                    new_mapper.weight[start:end, :] = weight if weight.size(0) <= new_cls else weight[start:upper_bound, :]
                    new_mapper.weight[list(removed_)] = 0.
                    module.mapper = new_mapper            

    def calculate_gate_norm(self):
        w1 = nn.utils.weight_norm(self.gate, name="weight")
        print(w1.weight_g)
        nn.utils.remove_weight_norm(w1)

    def bias_forward(self, task, output):
        """Modified version from FACIL"""
        return self.bias_layers[task](output)

    def freeze_previous_experts(self):
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_experts(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = False
            b.beta.requires_grad = False

    def unfreeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = True
            b.beta.requires_grad = True

    def set_gate(self, grad):
        for name, param in self.named_parameters():
            if name == "gate":
                param.requires_grad = grad

    def unfreeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = True

    def forward(self, x, task=None, train_step=2):
        gate_outputs = None
        if train_step == 1:            
            expert_outputs = self.experts[task](x)
        else:
            # if x.size(0) != 1:
            #     gate_outputs = self.relu(self.bn1(self.gate(x)))
            # else:
            #     gate_outputs = self.relu(self.in1(self.gate(x)))
            gate_outputs = self.gate(x)
            # gate_outputs = self.relu(self.gate(x))

            # gate_outputs = self.relu(self.gate(x))
            gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)
            
            expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
            # print(f"in new_expert 1: {len(expert_outputs)}")
            expert_outputs = torch.stack(expert_outputs, 1)
            # print(f"in new_expert 2: {expert_outputs.size()}")
            expert_outputs = gate_outputs_uns@expert_outputs
            # print(f"in new_expert 3: {expert_outputs.size()}")
            expert_outputs = torch.squeeze(expert_outputs, 1) # only squeeze the middle 1 dimension
            # print(f"in new_expert 4: {expert_outputs.size()}")

        return expert_outputs, gate_outputs

    def predict(self, x, task):
        expert_output = self.experts[task](x)
        return expert_output
        