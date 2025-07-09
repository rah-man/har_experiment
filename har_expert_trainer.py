import argparse
import base_har
import clusterlib
import copy
import numpy as np
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
# import wandb

from collections import Counter
from itertools import zip_longest
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.feature_extraction import create_feature_extractor

from earlystopping import EarlyStopping
from exemplar_selection import HerdingExemplarsSelector, RandomExemplarsSelector
from new_expert import DynamicExpert
from mets import Metrics2
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, classification_report, f1_score

class Trainer:
    def __init__(self, data, taskcla, clsorder, n_epochs, batch, in_features, 
                lr, mom, wd, step, total_exemplar=0, exemplar_per_class=10, 
                strategy=0, n_models=5, gmm=0, mix=0, kind='pamap', device=0, 
                stop=0, hidden_size=0, use_max=0, use_bias=1, n_components=[2, 3]):
        self.data = data
        self.taskcla = taskcla
        self.clsorder = clsorder        
        self.n_epochs = n_epochs
        self.batch = batch
        self.in_features = in_features
        self.lr = lr
        self.mom = mom
        self.wd = wd
        self.step = step
        self.strategy = strategy
        self.use_gmm = False if gmm == 0 else True
        self.use_mixup = False if mix == 0 else True
        self.total_exemplar = total_exemplar
        self.exemplar_per_class = exemplar_per_class
        self.kind = kind
        self.stop = stop
        self.exemplar_selector = None
        self.use_max = use_max
        self.pure_time = False   # True if we want to ignore logging processes hence only capture the pure train time
        self.use_bias = use_bias
        self.n_components = n_components
        
        if not self.use_gmm and (total_exemplar != 0 or exemplar_per_class != 0):
            self.exemplar_selector = RandomExemplarsSelector(total_exemplar=total_exemplar, exemplar_per_class=exemplar_per_class)
            # self.exemplar_selector = HerdingExemplarsSelector(total_exemplar=total_exemplar, exemplar_per_class=exemplar_per_class)
            # self.exemplar_selector = None
            
        """dynamic properties"""
        self.models = None  # initialised as Ensemble in self.expand_expert()
        self.optimiser = None   # updated in self.expand_expert()
        self.scheduler = None   # updated in self.expand_expert()
        self.clsgmm = {}        # to store class gmm
        
        """additional standard properties"""
        self.criterion = nn.CrossEntropyLoss()
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda:0' if device == 0 else f'cuda:{device}'
        # self.hidden_size = 91 if (kind == 'wisdm' or kind == 'wisdmflex') else int(1000 // (len(self.data) - 1))
        self.hidden_size = int(1000 // (len(self.data) - 1)) if hidden_size == 0 else hidden_size
        # if kind == 'wisdm':
        #     self.hidden_size = 91 
        self.metric = Metrics2()
        # CREATE DYNAMIC MoE model
        self.model = DynamicExpert(input_size=self.in_features, hidden_size=self.hidden_size)
        self.cls2idx = {v: k for k, v in enumerate(self.clsorder)}
        self.seen_cls = 0
        self.subclass_mapper = self.data[0]['ordered']
        self.previous_task_nums = []
        self.previous_task_labels = []
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 1
        self.alpha = 0.25
        self.gmm_pool = {}

        """logging properties"""
        self.task_gmm_creation = {}     # sampling time in format: {task: time}
        self.task_gmm_creation_wall = {}
        self.class_gmm_creation = {}    # sampling time in format: {class: time}
        self.class_gmm_creation_wall = {}
        self.task_gmm_learning = {}     # learning time in format: {task: time}
        self.task_gmm_learning_wall = {}
        self.class_gmm_learning = {}    # learning time in format: {class: [time]}
        self.class_gmm_learning_wall = {}
        self.test_confusion_matrix = {} # format: {'true': [], 'preds': [], 'labels': [], 0: {'true': [], 'preds': [], 'labels': []}}
        self.train_accuracy = {}        # format: {0: {expert: 0.5, gate: 0.5}, 1: {expert: 0.5, gate: 0.5}, ...}
        self.train_loss = {}            # format: {0: {expert: 0.5, gate: 0.5}, 1: {expert: 0.5, gate: 0.5}, ...}
        self.val_accuracy = {}          # format: {0: {expert: 0.5, gate: 0.5}, 1: {expert: 0.5, gate: 0.5}, ...}
        self.val_loss = {}              # format: {0: {expert: 0.5, gate: 0.5}, 1: {expert: 0.5, gate: 0.5}, ...}
        self.expert_train_time = {}     #
        self.expert_train_time_wall = {} 
        self.gate_train_time = {}       # 
        self.gate_train_time_wall = {}
        self.prediction_time = None     # format: floating nano seconds
        self.prediction_time_wall = None
        self.task_params = {}

    def update_clsgmm(self, clsgmm_):
        self.clsgmm = {**self.clsgmm, **clsgmm_}        

    def grouper(self, iterable, n, fillvalue=None):
        if isinstance(n, list):
            i = 0
            group = []
            for cp in n:
                group.append(tuple(iterable[i:i+cp]))
                i += cp
            return group
        else:
            args = [iter(iterable)] * n
            group = zip_longest(*args, fillvalue=fillvalue) 
            return group   

    def update_classmap(self, new_cls):
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}

    def expand_expert(self, seen_cls, new_cls):
        self.model.expand_expert(seen_cls, new_cls)
        if self.kind == 'wisdm':
            self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=self.step)      

    def build_gmm(self, task, n_components=[2, 3]):
        gmm_start = time.process_time()
        gmm_start_wall = time.time()
        self.update_clsgmm(clusterlib.class_gmm(self.data[task]['trn']['x'], self.data[task]['trn']['y'], self.class_gmm_learning, n_components))
        gmm_end = time.process_time()
        gmm_end_wall = time.time()

        self.task_gmm_learning[task] = gmm_end - gmm_start
        self.task_gmm_learning_wall[task] = gmm_end_wall - gmm_start_wall

        all = set(self.clsgmm.keys())
        prev = set(self.previous_task_labels)
        # print(f"Polling GMM for: {all.symmetric_difference(prev)}")
        for c in all.symmetric_difference(prev):
            # create 1000 or 100 pre-samples
            x = self.clsgmm[c].sample(1000)[0] if self.use_max else self.clsgmm[c].sample(100)[0]
            # x = self.clsgmm[c].sample(100)[0]   
            self.gmm_pool[c] = np.array(x)      

    def build_gmmbuffer(self, task):
        y_task = self.data[task]['trn']['y']                        
        if self.use_max:
            _, most_common_freq = Counter(y_task).most_common(1)[0] # using max samples in current task for gmm samples
            if most_common_freq > 1000:
                most_common_freq = 1000
        else:
            most_common_freq = self.exemplar_per_class # using k gmm samples
        buffer_x, buffer_y = [], []
        choices = [i for i in range(1000)] if self.use_max else [i for i in range(100)]

        # get random k from buffer
        task_start = time.process_time()
        task_start_wall = time.time()
        for k in self.clsgmm.keys():
            idx = np.random.choice(choices, size=most_common_freq, replace=False)
            buffer_x.extend(self.gmm_pool[k][idx].tolist())
            buffer_y.extend([k for _ in range(most_common_freq)])
        task_end = time.process_time()
        task_end_wall = time.time()
        self.task_gmm_creation[task] = task_end - task_start
        self.task_gmm_creation_wall[task] = task_end_wall - task_start_wall

        # """
        # use gmm sampling for all classes (old and new)
        # task_start = time.process_time()
        # for k, v in self.clsgmm.items():
        #     class_start = time.process_time()
        #     sample = v.sample(most_common_freq)[0]
        #     class_end = time.process_time()

        #     temp = self.class_gmm_creation.get(task, [])
        #     temp.append(class_end - class_start)
        #     self.class_gmm_creation[task] = temp

        #     buffer_x.extend(sample)
        #     buffer_y.extend([k for _ in range(most_common_freq)])
        # task_end = time.process_time()
        # self.task_gmm_creation[task] = task_end - task_start
        # print(f"\tGENERATING GMM SAMPLES: {Counter(buffer_y)}")

        # """

        """
        # generate gmm samples for old classes
        for old_class in self.previous_task_labels:
        # for k, v in self.clsgmm.items():
            buffer_x.extend(self.clsgmm[old_class].sample(most_common_freq)[0])  # sample() returns tuple(0) is the features, tuple(1) is the gaussian class
            buffer_y.extend([old_class for _ in range(most_common_freq)])
            # buffer_x.extend(v.sample(most_common_freq)[0])
            # buffer_y.extend([k for _ in range(most_common_freq)])
        print(f"\tGENERATING GMM SAMPLES FOR OLD CLASSES: {Counter(buffer_y)}")

        # add raw samples for current classes
        buffer_x.extend(self.data[task]['trn']['x'])
        buffer_y.extend(self.data[task]['trn']['y'])
        print(f"\tADDING RAW SAMPLES FOR CURRENT CLASSES: {Counter(self.data[task]['trn']['y'])}")
        """

        buffer_x = np.vstack(buffer_x).astype(np.float32)
        buffer_y = np.array(buffer_y).astype(np.int32)
        return buffer_x, buffer_y

    def update_logger(self, dict_, task, key, value):
        temp_ = dict_.get(task, {})
        temp_[key] = value
        dict_[task] = temp_

    def train_loop(self, steps=2):                
        val_loaders = []        
        final_train_batch = []
        num_cls = 0
        just_expert = False
        for task in range(len(self.data.keys()) - 1):
            if self.stop:
                early_stop = EarlyStopping(verbose=False)
            
            # If this is not the initial task, save before updating the expert in case needed for distillation
            if task > 0:
                self.previous_model = copy.deepcopy(self.model)
                self.previous_model.to(self.device)
                self.previous_model.eval()

            # update the model for new classes in each iteration
            new_cls = len(self.data[task]["classes"])
            # self.update_classmap(self.data[task]["classes"])
            
            # EXPAND EXISTING EXPERT WITH INCOMING TASK
            self.expand_expert(self.seen_cls, new_cls)
            # print(self.model)

            # log task parameters
            total_parameters = sum(param.numel() for param in self.model.parameters())
            self.task_params[task] = total_parameters

            # print("=====GATE WEIGHT BEGIN=====")
            # self.model.calculate_gate_norm()
            # print("==========================")            

            self.model.to(self.device)   
            
            # build/update self.clsgmm here
            if self.use_gmm: 
                self.build_gmm(task, self.n_components)
                # print(f"task: {task}\tgmm.keys(): {self.clsgmm.keys()}")
                # print(f"task: {task}\tprevious_task_labels: {self.previous_task_labels}")
                
            # load the dataset for current task
            train_dataloader = DataLoader(base_har.BaseDataset(self.data[task]['trn']), batch_size=self.batch, shuffle=True)
            val_dataloader = DataLoader(base_har.BaseDataset(self.data[task]['val']), batch_size=self.batch, shuffle=True)
            val_loaders.append(val_dataloader)            

            self.model.freeze_previous_experts()
            self.model.freeze_all_bias()
            self.model.set_gate(False)
            # print(self.model) 

            train_loss = []
            train_acc = []

            num_cls += self.data[task]['ncla']

            train_start = time.process_time()
            train_start_wall = time.time()
            # START TRAINING STEP-1, i.e., train individual expert
            for epoch in range(self.n_epochs):
                ypreds, ytrue = [], []
                self.model.train()

                running_train_loss = [] # store train_loss per epoch
                dataset_len = 0
                pred_correct = 0.0

                for x, y in train_dataloader:
                    x = x.to(self.device)
                    y = y.type(torch.LongTensor).to(self.device)
                    outputs, gate_outputs = self.model(x, task, train_step=1)    # gate_outputs will be None for train_step=1
                    
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                                  
                    if not self.pure_time:
                        # remove only if we want to log the prediction
                        predicted = torch.argmax(outputs, 1)
                        pred_correct += predicted.eq(y).cpu().sum().float()

                        running_train_loss.append(loss.item())
                        dataset_len += x.size(0) #+ inputs.size(0)
                
                if self.stop:
                    early_stop(loss, self.model)

                if not self.pure_time:
                    train_loss.append(np.average(running_train_loss))
                    train_acc.append(100 * pred_correct.item() / dataset_len)
                if (epoch + 1) % 10 == 0:
                    print(f"STEP-1\tEpoch: {epoch+1}/{self.n_epochs}\tloss: {train_loss[-1]:.4f}\tstep1_train_accuracy: {train_acc[-1]:.4f}")
                if self.stop and early_stop.early_stop:
                    print(f"Early stopping. Exit epoch {epoch+1}")
                    break

            train_end = time.process_time()
            train_end_wall = time.time()
            self.expert_train_time[task] = train_end - train_start
            self.expert_train_time_wall[task] = train_end_wall - train_start_wall

            """ FOR LOGGER: RUN FINAL PREDICTION/LOSS FOR TRAIN SET """
            self.model.eval()
            x_temp = torch.tensor(np.array(self.data[task]['trn']['x'])).to(self.device)
            y_temp = torch.tensor(self.data[task]['trn']['y']).type(torch.LongTensor).to(self.device)
            with torch.no_grad():
                outs, _ = self.model(x_temp, task, train_step=1)
            pred_ = torch.argmax(outs, 1)
            acc_ = 100 * (pred_.eq(y_temp).cpu().sum().float() / y_temp.size(0)).cpu().item()
            loss_ = self.criterion(outs, y_temp).cpu().item()

            self.update_logger(self.train_accuracy, task, 'step1', acc_)
            self.update_logger(self.train_accuracy, task, 'step1_detail', train_acc)
            self.update_logger(self.train_loss, task, 'step1', loss_)
            self.update_logger(self.train_loss, task, 'step1_detail', train_loss)
            self.model.train()

            print("FINISH STEP 1")
            # print(self.model)
            # print(train_acc)
            # print(f'train time: {self.expert_train_time[task]}')
            # exit()
            
            # if self.exemplar_selector and task < len(self.data.keys()) - 1 and not self.use_gmm:
            if self.exemplar_selector and task < len(self.data.keys()) - 1 and not self.use_gmm:
                # print(f"\n=====RANDOM SELECTION FOR BUFFER=====")
                # print(f"\n=====HERDING SELECTION FOR BUFFER=====")
                # Use train and val data to be selected
                self.exemplar_selector.update_buffer(self.data[task]['trn']['x'], self.data[task]['trn']['y'], num_cls)
                print(f"Buffer size: {Counter(self.exemplar_selector.buffer_y)}\n")  

            # TRAIN THE GATE
            if not just_expert and steps == 2:
                print(f"Task-{task+1}\tSTARTING STEP 2")
                self.model.freeze_all_experts()                
                if self.use_bias:
                    self.model.unfreeze_all_bias()
                self.model.set_gate(True)

                if self.exemplar_selector and self.exemplar_selector.buffer_x and not self.use_gmm:
                    buffer_x, buffer_y = self.exemplar_selector.buffer_x, self.exemplar_selector.buffer_y
                elif self.use_gmm:
                    buffer_x, buffer_y = self.build_gmmbuffer(task)

                gate_labels = np.vectorize(self.subclass_mapper.get)(buffer_y)
                train_dataloader = DataLoader(base_har.BaseDataset({'x': buffer_x, 'y': buffer_y, 'gate_label': gate_labels}), batch_size=self.batch, shuffle=True)
                print(f"CLASS COUNTER: {Counter(buffer_y)}")

                # class_distribution = Counter(np.vectorize(self.cls2idx.get)(train_dataloader.dataset.labels).tolist())
                # class_distribution = [v for _, v in class_distribution.items()]
                
                train_loss = [] # store train_loss per epoch                
                gate_loss_ = [] # store gate_loss per epoch
                
                train_acc_d = []
                gate_acc_d = []

                gate_optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.wd)                                

                if self.stop:
                    early_stop = EarlyStopping(verbose=False)

                gate_start = time.process_time()
                gate_start_wall = time.time()
                for epoch in range(self.n_epochs):
                # for epoch in range(200):                    
                    running_train_loss, running_gate_loss = [], []
                    dataset_len, gate_correct, pred_correct = 0, 0.0, 0.0
                    ypreds, ytrue = [], []
                    
                    for i, (x, y, gate_labels) in enumerate(train_dataloader):
                        x = x.to(self.device)
                        y = y.type(torch.LongTensor).to(self.device)
                        gate_labels = gate_labels.to(self.device)
                        
                        outputs, gate_outputs = self.model(x, train_step=2)
                            
                        if task > 0 and self.use_bias:
                            # _, old_gate_outputs = self.previous_model(x, train_step=2)

                            bias_outputs = []
                            prev = 0
                            for i, num_class in enumerate(self.previous_task_nums):
                                outs = outputs[:, prev:(prev+num_class)]
                                bias_outputs.append(self.model.bias_forward(task, outs))
                                prev += num_class
                            old_cls_outputs = torch.cat(bias_outputs, dim=1)
                            new_cls_outputs = self.model.bias_forward(task, outputs[:, prev:])  # prev should point to the last task
                            pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                            loss = self.criterion(pred_all_classes, y)
                            loss += 0.1 * ((self.model.bias_layers[task].beta[0] ** 2) / 2)
                        else:
                            if len(outputs.size()) == 1:
                                outputs = outputs.view(-1, 1)
                            loss = self.criterion(outputs, y)

                        # this should be the default case, i.e. there's no other way to train the gate without this if-block
                        gate_loss = self.criterion(gate_outputs, gate_labels)
                        loss += gate_loss
                        loss.backward()

                        if not self.pure_time:
                            running_train_loss.append(loss.item())
                            running_gate_loss.append(gate_loss.item())
                            gate_preds = torch.argmax(gate_outputs.data, 1)
                            gate_correct += gate_preds.eq(gate_labels).cpu().sum().float()
                                                
                        
                        # value clipping
                        # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

                        # norm clipping
                        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
                        
                        gate_optimiser.step()
                        gate_optimiser.zero_grad()

                        if not self.pure_time:
                            predicted = torch.argmax(outputs.data, 1) if (task == 0 or not self.use_bias) else torch.argmax(pred_all_classes, 1)                        
                            pred_correct += predicted.eq(y).cpu().sum().float()

                            ypreds.extend(predicted.detach().cpu().tolist())
                            ytrue.extend(y.detach().cpu().tolist())
                            
                            dataset_len += x.size(0) #+ inputs.size(0)

                    # gate_scheduler.step()   

                    if self.stop:
                        early_stop(loss, self.model)

                    if not self.pure_time:
                        train_loss.append(np.average(running_train_loss))
                        gate_loss_.append(np.average(running_gate_loss))

                        train_acc_d.append(100 * pred_correct.item() / dataset_len)
                        gate_acc_d.append(100 * gate_correct.item() / dataset_len)

                    if (epoch + 1) % 50 == 0:
                        print(f"STEP-2\tEpoch: {epoch+1}/200\tclassification_loss: {train_loss[-1]:.4f}\tgate_loss: {gate_loss_[-1]:.4f}\tstep2_classification_accuracy: {train_acc_d[-1]:.4f}\tstep_2_gate_accuracy: {gate_acc_d[-1]:.4f}")
                    
                    if self.stop and early_stop.early_stop:
                        print(f"Early stopping. Exit epoch {epoch+1}")
                        break                                        
                
                gate_end = time.process_time()
                gate_end_wall = time.time()
                self.gate_train_time[task] = gate_end - gate_start
                self.gate_train_time_wall[task] = gate_end_wall - gate_start_wall

                print("FINISH STEP 2")       
                print()

            """ FOR LOGGER: RUN FINAL PREDICTION/LOSS FOR TRAIN SET """
            self.model.eval()
            x_temp = torch.tensor(np.array(buffer_x)).to(self.device)
            y_temp = torch.tensor(buffer_y).type(torch.LongTensor).to(self.device)
            gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y_temp.cpu().numpy())).type(torch.LongTensor).to(self.device)
            with torch.no_grad():
                outputs, gate_outs = self.model(x_temp, train_step=2)

            if task > 0 and self.use_bias:
                bias_outputs = []
                prev = 0
                for i, num_class in enumerate(self.previous_task_nums):
                    outs = outputs[:, prev:(prev+num_class)]
                    bias_outputs.append(self.model.bias_forward(task, outs))
                    prev += num_class
                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                new_cls_outputs = self.model.bias_forward(task, outputs[:, prev:])
                pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                predicted = torch.argmax(pred_all_classes, 1)
            else:                         
                if len(outputs.size()) == 1:
                    outputs = outputs.view(-1, 1)
                pred_all_classes = outputs
                predicted = torch.argmax(outputs, 1)

            classifier_loss =  self.criterion(pred_all_classes, y_temp)
            classifier_acc = 100 * (predicted.eq(y_temp).cpu().sum().float() / y_temp.size(0)).cpu().item()

            gate_pred_ = torch.argmax(gate_outs.data, 1)
            acc_ = 100 * (gate_pred_.eq(gate_labels).cpu().sum().float() / y_temp.size(0)).cpu().item()
            gate_loss = self.criterion(gate_outs, gate_labels).cpu().item()

            self.update_logger(self.train_accuracy, task, 'step2_gate', acc_)
            self.update_logger(self.train_loss, task, 'step2_gate', gate_loss)
            self.update_logger(self.train_accuracy, task, 'step2_gate_detail', gate_acc_d)
            self.update_logger(self.train_loss, task, 'step2_gate_detail', gate_loss_)  

            self.update_logger(self.train_accuracy, task, 'step2_classifier', classifier_acc)
            self.update_logger(self.train_loss, task, 'step2_classifier', classifier_loss.item())

            self.update_logger(self.train_accuracy, task, 'step2_classifier_detail', train_acc_d)
            self.update_logger(self.train_loss, task, 'step2_classifier_detail', train_loss)              

            # self.update_logger(self.train_accuracy, 'gate_detail', )

            self.model.train()

            self.model.unfreeze_all()

            print(f"VALIDATING MODEL")
            if not just_expert and val_loaders:                
                self.model.eval()
                val_loss_ = []
                all_y, all_preds, all_gate, all_gate_preds = [], [], [], []
                logit_preds, logit_gate = [], []
                for nt, valloader in enumerate(val_loaders):
                    
                    running_val_loss = []
                    dataset_len = 0
                    gate_correct = 0
                    ypreds, ytrue = [], []
                    for i, (x, y) in enumerate(valloader):
                        x = x.to(self.device)
                        y = y.type(torch.LongTensor).to(self.device)

                        with torch.no_grad():
                            outputs, gate_outputs = self.model(x, train_step=2)

                            if task > 0 and self.use_bias:
                                # when there is only a sample in a batch
                                if len(outputs.size()) == 1:                                    
                                    outputs = outputs.unsqueeze(0)
                                bias_outputs = []
                                prev = 0
                                for i, num_class in enumerate(self.previous_task_nums):
                                    outs = outputs[:, prev:(prev+num_class)]
                                    bias_outputs.append(self.model.bias_forward(task, outs))
                                    # bias_outputs.append(self.model.bias_forward(i, outs))
                                    prev += num_class
                                old_cls_outputs = torch.cat(bias_outputs, dim=1)
                                new_cls_outputs = self.model.bias_forward(task, outputs[:, prev:])
                                pred_all_classes = torch.cat([old_cls_outputs, new_cls_outputs], dim=1)
                                predicted = torch.argmax(pred_all_classes, 1)
                            else:                         
                                if len(outputs.size()) == 1:
                                    outputs = outputs.view(-1, 1)
                                pred_all_classes = outputs
                                predicted = torch.argmax(outputs, 1)

                            logit_preds.extend(pred_all_classes)
                            logit_gate.extend(gate_outputs)

                            if self.subclass_mapper:
                                gate_labels = torch.tensor(np.vectorize(self.subclass_mapper.get)(y.cpu().numpy())).type(torch.LongTensor).to(self.device)
                            #     gate_loss = self.criterion(gate_outputs, gate_labels)
                            #     loss += gate_loss

                                gate_preds = torch.argmax(gate_outputs.data, 1)
                                gate_correct += (gate_preds == gate_labels).sum().item()

                                all_gate.extend(gate_labels.cpu().tolist())
                                all_gate_preds.extend(gate_preds.cpu().tolist())
                                

                            # running_val_loss.append(loss.item())
                            dataset_len += x.size(0)

                        # predicted = torch.argmax(outputs.data, 1)
                        # print(f"\tx.size(): {x.size()}\tpredicted.size(): {predicted.size()}\ty.size(): {y.size()}")
                        pred_ = predicted.detach().cpu().tolist()
                        y_ = y.detach().cpu().tolist()

                        ypreds.extend(pred_)
                        ytrue.extend(y_)

                        all_y.extend(y_)
                        all_preds.extend(pred_)

                    # val_loss_.append(np.average(running_val_loss))
                    task_accuracy = 100 * accuracy_score(ytrue, ypreds)
                    # print(f"\tTask-{nt}\tval_loss: {val_loss_[-1]:.4f}\tval_accuracy: {task_accuracy:.4f}\tgate_accuracy: {100 * (gate_correct / dataset_len):.4f}")
                    print(f"\tTask-{nt}\tval_accuracy: {task_accuracy:.4f}\tgate_accuracy: {100 * (gate_correct / dataset_len):.4f}")

                    if self.metric:
                        self.metric.add_accuracy(task, task_accuracy)
                    # self.val_loss.append(val_loss_)

                """ FOR VALIDATION LOGGING """            
                all_y = torch.tensor(all_y).to(self.device)
                all_preds = torch.tensor(all_preds).to(self.device)
                all_gate = torch.tensor(all_gate).to(self.device)
                all_gate_preds = torch.tensor(all_gate_preds).to(self.device)

                logit_preds = torch.stack(logit_preds).to(self.device)
                logit_gate = torch.stack(logit_gate).to(self.device)                

                acc_ = 100 * (all_preds.eq(all_y).cpu().sum().float() / all_y.size(0)).cpu().item()
                gate_acc = 100 * (all_gate_preds.eq(all_gate).cpu().sum().float() / all_gate.size(0)).cpu().item()

                loss_ = self.criterion(logit_preds, all_y).cpu().item()
                gate_loss_ = self.criterion(logit_gate, all_gate).to(self.device).cpu().item()

                self.update_logger(self.val_accuracy, task, 'expert', acc_)
                self.update_logger(self.val_accuracy, task, 'gate', gate_acc)

                self.update_logger(self.val_loss, task, 'expert', loss_)
                self.update_logger(self.val_loss, task, 'gate', gate_loss_)

                print(f"\tGATE_ACCURACY: {gate_acc:.4f}")
                print()
            print()

            self.seen_cls += new_cls
            self.previous_task_nums.append(self.data[task]["ncla"])
            self.previous_task_labels.extend(self.data[task]["classes"])

        # print(f"\nTRAIN_ACCURACY: {self.train_accuracy}\n")
        # print(f"\nTRAIN_LOSS: {self.train_loss}\n")
        # print(f"\nVAL_ACCURACY: {self.val_accuracy}\n")
        # print(f"\nVAL_LOSS: {self.val_loss}\n")
        
        print()

    def test(self):
        # print(self.model)
        self.model.to(self.device)
        self.model.eval()

        print("CALCULATING TEST ACCURACY PER TASK")
        all_preds, all_true = [], []   
        for task in range(len(self.data.keys()) - 1):
            ypreds, ytrue = [], []
            x_test, y_test = self.data[task]["tst"]["x"], self.data[task]["tst"]["y"]
            
            x = torch.Tensor(x_test).to(self.device)
            y = torch.Tensor(y_test).type(torch.LongTensor).to(self.device)

            with torch.no_grad():
                outputs, gate_outputs = self.model(x)

                # when there is a single sample in a batch
                if len(outputs.size()) == 1:
                    outputs = outputs.unsqueeze(0)
                    
                if self.use_bias:
                    bias_outputs = []
                    prev = 0
                    for i, num_class in enumerate(self.previous_task_nums):
                        outs = outputs[:, prev:(prev+num_class)]
                        bias_outputs.append(self.model.bias_forward(task, outs))
                        # bias_outputs.append(self.model.bias_forward(i, outs))
                        prev += num_class
                    pred_all_classes = torch.cat(bias_outputs, dim=1)
                    # predicted = torch.argmax(pred_all_classes.data, 1)
                
                # predicted = torch.argmax(outputs.data, 1) if not self.model.bias_layers else torch.argmax(pred_all_classes.data, 1)
                predicted = torch.argmax(outputs.data, 1) if not self.use_bias else torch.argmax(pred_all_classes.data, 1)
                true_ = y.detach().cpu().tolist()
                preds_ = predicted.detach().cpu().tolist()
                self.test_confusion_matrix[task] = {'true': true_, 'preds': preds_, 'labels': sorted(np.unique(true_))}

                ypreds.extend(preds_)
                ytrue.extend(true_)
                
                all_preds.extend(preds_)
                all_true.extend(true_)

            cls_ = np.unique(self.data[task]['classes'])
            # print(f"\t{np.unique(ytrue)}\t{np.unique(ypreds)}")
            print(f"\tTASK-{task}\tCLASSES: {cls_}\ttest_accuracy: {(100 * accuracy_score(ytrue, ypreds)):.4f}")

        # pick one test sample and pass to the whole model to get the prediction time
        x_ = torch.unsqueeze(x[-1], 0)
        pred_start = time.process_time()
        pred_start_wall = time.time()
        with torch.no_grad():
            outputs, _ = self.model(x_)
            outputs = outputs.unsqueeze(0)
        if self.use_bias:
            bias_outputs = []
            prev = 0
            for i, num_class in enumerate(self.previous_task_nums):
                outs = outputs[:, prev:(prev+num_class)]
                bias_outputs.append(self.model.bias_forward(task, outs))
                # bias_outputs.append(self.model.bias_forward(i, outs))
                prev += num_class
            pred_all_classes = torch.cat(bias_outputs, dim=1)
            # predicted = torch.argmax(pred_all_classes.data, 1)
        predicted = torch.argmax(outputs.data, 1) if not self.use_bias else torch.argmax(pred_all_classes.data, 1)
        pred_end = time.process_time()
        pred_end_wall = time.time()
        print(f"PREDICTION TIME: {pred_end - pred_start}")
        self.prediction_time = pred_end - pred_start
        self.prediction_time_wall = pred_end_wall - pred_start_wall

        self.test_confusion_matrix['true'] = all_true
        self.test_confusion_matrix['preds'] = all_preds
        self.test_confusion_matrix['labels'] = sorted(np.unique(all_true))

        print("\n====================\n")
        print(f"f1_score(micro): {100 * f1_score(all_true, all_preds, average='micro', zero_division=1)}")
        print(f"f1_score(macro): {100 * f1_score(all_true, all_preds, average='macro', zero_division=1)}")
        print(f"Classification report:\n{classification_report(all_true, all_preds, zero_division=1)}")

    def test2(self):
        self.model.to(self.device)
        self.model.eval()

        x, y = [], []
        for task in range(len(self.data.keys()) - 1):
            x.extend(self.data[task]["tst"]["x"])
            y.extend(self.data[task]["tst"]["y"])            

        x = torch.Tensor(np.array(x)).to(self.device)
        # y = torch.Tensor(y).type(torch.LongTensor).to(self.device)

        with torch.no_grad():
            outputs, gate_outputs = self.model(x)

        bias_outputs = []
        prev = 0
        for i, num_class in enumerate(self.previous_task_nums):
            outs = outputs[:, prev:(prev+num_class)]
            bias_outputs.append(self.model.bias_forward(i, outs))
            prev += num_class

        pred_all_classes = torch.cat(bias_outputs, dim=1)
        predicted = torch.argmax(pred_all_classes.data, 1).detach().cpu().tolist()
        # y = y.detach().cpu().tolist()

        gate_preds = torch.argmax(gate_outputs.data, 1).cpu().tolist()
        gate_labels = np.vectorize(self.subclass_mapper.get)(y)

        print("\n====================\n")
        print(f"f1_score(micro): {100 * f1_score(y, predicted, average='micro', zero_division=1)}")
        print(f"f1_score(macro): {100 * f1_score(y, predicted, average='macro', zero_division=1)}")
        print(f"gate_accuracy: {100 * accuracy_score(gate_labels, gate_preds)}")
        print(f"Classification report:\n{classification_report(y, predicted, zero_division=1)}")


    def reduce_learning_rate(self, optimiser):        
        for param_group in optimiser.param_groups:
            old_lr = param_group["lr"]
        for param_group in optimiser.param_groups:
            param_group["lr"] = param_group["lr"] / 10
            new_lr = param_group["lr"]
        print(f"Reduce learning_rate from {old_lr} to {new_lr}")

    def print_params(self):
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

    def save_model(self, path):
        x_test, y_test = [], []
        for task in range(len(self.data.keys()) - 1):
            x_test.extend(self.data[task]["tst"]["x"])
            y_test.extend(self.data[task]["tst"]["y"])
        to_save = {"model": self.model, "gate_mapper": self.subclass_mapper, "cls2idx": self.cls2idx, "clsgmm": self.clsgmm, "test_data": {"x": x_test, "y": y_test}}
        torch.save(to_save, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, description='Which dataset to train? 0: pamap, 1: dsads, 2: hapt, 3: wisdm')
    parser.add_argument("-lr", type=float, description='Learning rate')
    parser.add_argument("-wd", type=float, description='Weight decay')
    parser.add_argument("-mom", type=float, description='Momentum')
    parser.add_argument("-b", type=int, description='Batch size')
    parser.add_argument("-sch", type=int, description='Scheduler step')
    # parser.add_argument("-e", type=int)
    parser.add_argument('-total_exemplar', default=0, type=int)
    parser.add_argument("-exemplar_per_class", default=0, type=int)
    parser.add_argument('-strategy', default=0, type=int)
    parser.add_argument('-n_models', default=5, type=int)
    parser.add_argument('-model_path', default='', type=str, description='Where to save the model')
    parser.add_argument('-gmm', default=0, type=int)
    parser.add_argument('-mix', default=0, type=int)
    parser.add_argument('-save_path', default='', type=str)
    parser.add_argument('-cud', default=0, type=int)
    parser.add_argument('-load_data', default='', type=str)
    parser.add_argument('-stop', default=0, type=int)
    parser.add_argument('-hidden_size', default=0, type=int)
    parser.add_argument('-use_max', default=0, type=int)
    parser.add_argument('-use_bias', default=1, type=int)
    parser.add_argument('-cust', default=0, type=int)
    parser.add_argument('-ncomp', nargs='*', type=int, default=[2, 3])

    args = parser.parse_args()

    d = args.d
    lr = args.lr
    wd = args.wd
    mom = args.mom
    batch = args.b
    n_epochs = args.e
    scheduler_step = args.sch
    total_exemplar = args.total_exemplar
    exemplar_per_class = args.exemplar_per_class
    strategy = args.strategy
    n_models = args.n_models
    model_path = args.model_path
    gmm = args.gmm
    mix = args.mix
    save_path = args.save_path
    device = args.cud
    load_data = args.load_data
    stop = args.stop
    hidden_size = args.hidden_size
    use_max = args.use_max
    use_bias = args.use_bias
    cust = args.cust
    n_components = args.ncomp
    
    path = ["/home/fr27/Documents/pyscript/har/DSADS/pamap.mat", 
            "/home/fr27/Documents/pyscript/har/DSADS/dsads.mat", 
            "/home/fr27/Documents/pyscript/har/HAPT", 
            "/home/fr27/Documents/pyscript/wisdm/dataset/arff_files/phone/accel/all.csv", 
            "/home/fr27/Documents/pyscript/har/DSADS/dsads.mat",
            "/home/fr27/Documents/pyscript/har/DSADS/dsads.mat",
            "/home/fr27/Documents/pyscript/wisdm/dataset/arff_files/phone/accel/all.csv", ]
    kind = ["pamap", "dsads", "hapt", "wisdm", "flex", "flex2", "wisdmflex"]
    num_classes = [12, 19, 12, 18, 152, 95, 894]
    in_features = [243, 405, 561, 91, 405, 405, 91]    

    custom_class = [None, [3, 3, 4, 5, 4]]

    if load_data:
        # if the data has been saved and provided as a pickle file
        # Note: this option is useful for having the same train/test split
        if isinstance(pickle.load(open(load_data, 'rb'))['data'], dict):
            data = pickle.load(open(load_data, 'rb'))['data']
            clsorder = pickle.load(open(load_data, 'rb'))['clsorder']
            taskcla = None
        else:
            data, taskcla, clsorder = pickle.load(open(load_data, 'rb'))['data']
    else:
        # load the dataset
        data, taskcla, clsorder = base_har.get_data(path[d], kind=kind[d], save_path=save_path, custom_class=custom_class[cust])

    cls2idx = {v: k for k, v in enumerate(clsorder)}    # NOT NEEDED FOR CURRENT TRAIN/VAL/TEST DATASET/LOADER, ALREADY MAPPED. NEEDED FOR EXTERNAL DATASET FOR TESTING

    walltime_start, processtime_start = time.time(), time.process_time()
    trainer = Trainer(data, taskcla, clsorder, n_epochs, batch, 
                    in_features[d], lr, mom, wd, scheduler_step, 
                    total_exemplar, exemplar_per_class, strategy, 
                    n_models, gmm, mix, kind[d], device, stop, 
                    hidden_size, use_max, use_bias, n_components=n_components)
    trainer.train_loop()
    walltime_end, processtime_end = time.time(), time.process_time()
    elapsed_walltime = walltime_end - walltime_start
    elapsed_processtime = processtime_end - processtime_start
    print('Execution time:', )
    print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    # exit()

    faa = trainer.metric.final_average_accuracy()
    ff = trainer.metric.final_forgetting()
    print(f"FAA: {faa}")
    print(f"FF: {ff}")
    print()
    print("TRAINER.METRIC.ACCURACY")
    for k, v in trainer.metric.accuracy.items():
        print(f"{k}: {v}")
    # print()
    # print(trainer.metric.accuracy)
    # print("TRAINER.METRIC.FORGET")
    # for k, v in trainer.metric.forget.items():
    #     print(f"{k}: {v}")
    # print(trainer.metric.forget)
    print()
    print("=====RUNNING ON TEST SET=====")
    trainer.test()
    # trainer.test2()

    # print(trainer.model)

    # total_parameters = sum(param.numel() for param in trainer.model.parameters())
    # print(f'Parameters: {total_parameters}')    
    # exit()

    # if model_path:
    #     trainer.save_model(model_path)

    # losses = {"train_loss1": train_loss1, "train_loss2": train_loss2, "val_loss": val_loss, "metric_acc": met.accuracy, "metric_forget": met.forget}
    # pickle.dump(losses, open(pickle_file, "wb"))

    # model_path = pickle_file.split(".")[0]
    # torch.save(model, f"{model_path}.pth")
    
    # to_save = {"data": data, "task_cla": task_cla, "class_order": class_order}
    # pickle.dump(to_save, open(f"{model_path}.dat", "wb"))

    total_parameters = sum(param.numel() for param in trainer.model.parameters())

    data_ = {'data': data, 'taskcla': taskcla, 'clsorder': clsorder}

    # if save_path argument is provided
    to_save = {
        'task_gmm_sampling': trainer.task_gmm_creation,
        'class_gmm_sampling': trainer.class_gmm_creation,
        'task_gmm_learning': trainer.task_gmm_learning,
        'class_gmm_learning': trainer.class_gmm_learning,
        'test_confusion_matrix': trainer.test_confusion_matrix,
        'train_accuracy': trainer.train_accuracy,
        'train_loss': trainer.train_loss,
        'val_accuracy': trainer.val_accuracy,
        'val_loss': trainer.val_loss,
        'metric': trainer.metric.accuracy,
        'expert_train_time': trainer.expert_train_time,
        'gate_train_time': trainer.gate_train_time,
        'prediction_time': trainer.prediction_time,
        'total_parameters': total_parameters,
        'task_gmm_sampling_wall': trainer.task_gmm_creation_wall,
        'class_gmm_sampling_wall': trainer.class_gmm_creation_wall,
        'task_gmm_learning_wall': trainer.task_gmm_learning_wall,
        'class_gmm_learning_wall': trainer.class_gmm_learning_wall,
        'expert_train_time_wall': trainer.expert_train_time_wall,
        'gate_train_time_wall': trainer.gate_train_time_wall,
        'prediction_time_wall': trainer.prediction_time_wall,
        'task_parameters': trainer.task_params,
        'data': data_,
    }

    # print(f"expert_train_time: {trainer.expert_train_time}")
    # print(f"gate_train_time: {trainer.gate_train_time}")
    # print(f"prediction_time: {trainer.prediction_time}")
    # print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    """ not needed anymore as the saved data can be reused now """
    # to_save['data'] = [data, taskcla, clsorder]

    if save_path:
        pickle.dump(to_save, open(save_path, 'wb'))

    
    print(f'Parameters: {total_parameters}')
    print(f'Task parameters: {trainer.task_params}')

    # print(trainer.gate_train_time)
    # print(to_save)
    # print(trainer.model)

    # python ../har_expert_trainer.py -d 1 -lr 0.01 -wd 1e-4 -mom 0 -b 20 -e 200 -sch 50 -exemplar_per_class 10 -gmm 1 -model_path "for_paper/dsads_trial2/dsads_ns_10_1.pkl" > "for_paper/dsads_trial2/dsads_ns_10_1.txt"