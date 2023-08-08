# imports
import snntorch as snn
import torch
import torch.nn as nn

from copy import deepcopy

from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel


class MNIST(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        
        
        self.fc_layers = []
        input_size = 784
        num_hidden = model_params["num_hidden"]
        num_outputs = model_params["num_outputs"]
        beta = model_params["beta"]
        self.dtype = torch.float32
        self.num_steps = model_params["num_steps"]
        
        # Initialize layers
        self.fc1 = nn.Linear(input_size, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
            
    def forward(self, inputs):
        
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            # breakpoint()
            cur1 = self.fc1(inputs)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        
        
class MNISTModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        
        model_params = deepcopy(params["model"])
        self.model = self.build_model(model_params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device =  device #model_params["device"]
        
        compute_eval_metrics = model_params.get("compute_eval_metrics", [])
        if isinstance(compute_eval_metrics, bool) and compute_eval_metrics:
            compute_eval_metrics = ["accuracy"]  # All metrics
        
        self.accuracy_metric = None
        for name in compute_eval_metrics:
            if "accuracy" in name:
                from modelzoo.common.pytorch.metrics import AccuracyMetric
                self.accuracy_metric = AccuracyMetric(name=name)
                
        super().__init__(params=params, model=self.model, device=device)
            
    def build_model(self, model_params):
        dtype = torch.float32
        model = MNIST(model_params)
        for param in model.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)
        model.to(dtype)
        return model
    
    def __call__(self, data):
        inputs, labels = data
        inputs = inputs.view(inputs.shape[0], -1)
        
        spks, mems = self.model(inputs)
        
        # breakpoint()
        # Loss calculation for n_steps
        loss_val = torch.zeros((1), dtype=self.model.dtype, device=self.device)
        
        for step in range(self.model.num_steps):
            loss_val += self.loss_fn(mems[step], labels)
        
        # Accuracy
        _, predicted = spks.sum(dim=0).max(1)
        if self.accuracy_metric:
            labels = labels.clone()
            self.accuracy_metric(labels=labels, predictions=predicted)
        
        return loss_val
