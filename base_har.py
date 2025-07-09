import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy.io
import torch
import copy
import numpy as np
import torchvision.models as models
import random

from collections import Counter
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor

def split_df(df, user_col, exclude=[]):
    inside = df[~df.iloc[:, user_col].isin(exclude)]
    outside = df[df.iloc[:, user_col].isin(exclude)]
    return inside, outside

def drop_cols(df, cols=[]):
    return df.drop(columns=cols)

def to_tensor(df, class_col, scale=False, scaler=None):
    # data = {
    #     'data': [
    #         [
    #             768_dimension_of_ViT_embedding,
    #             the_label
    #         ],
    #         [],
    #         ...
    #     ],
    #     'targets': labels/classes as a whole
    # }
    X = df.iloc[:, :class_col].to_numpy()
    y = df.iloc[:, class_col].to_numpy()

    if scaler:
        X = scaler.transform(X)
    if scale:
        sc = StandardScaler()
        X = sc.fit_transform(X)

    y_map = {v: k for k, v in enumerate(np.unique(y))}
    y = torch.tensor(np.vectorize(y_map.get)(y)).type(torch.int)
    data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(X, y)]
    
    if scale:
        return {'data': data, 'targets': y, 'scaler': sc}
    return {'data': data, 'targets': y}

def data_dict(df, class_col):
    """ The same implementation as to_tensor() but the features are in numpy"""
    # data = {
    #     'data': [
    #         [
    #             har_features,
    #             the_label
    #         ],
    #         [],
    #         ...
    #     ],
    #     'targets': labels/classes as a whole
    # }
    X = df.iloc[:, :class_col].to_numpy()
    y = df.iloc[:, class_col].to_numpy()
    y_map = {v: k for k, v in enumerate(np.unique(y))}
    y = np.vectorize(y_map.get)(y)
    data = [[x_, y_] for x_, y_ in zip(X, y)]
    return {'data': data, 'targets': y}    

def make(df, user_col, exclude, drop=None, scale=False):
    inside, outside = split_df(df, user_col, exclude)
    inside = drop_cols(inside, [user_col])
    outside = drop_cols(outside, [user_col])

    if drop:
        inside = drop_cols(inside, drop)
        outside = drop_cols(outside, drop)

    if scale:
        train_data = to_tensor(inside, len(inside.columns)-1, scale)
        test_data = to_tensor(outside, len(inside.columns)-1, scaler=train_data['scaler'])
        del train_data['scaler']
    else:
        train_data = to_tensor(inside, len(inside.columns)-1)
        test_data = to_tensor(outside, len(inside.columns)-1)

    return train_data, test_data

def split_agents(df, agent_col, class_col, choice=2):
    agents = df.iloc[:, agent_col].unique()
    classes = df.iloc[:, class_col].unique()

    test_agent = np.random.choice(agents, choice, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    inside, outside = split_df(df, agent_col, test_agent)
    
    percentage = outside.shape[0]/df.shape[0]
    all_equal = np.array_equal(outside.iloc[:, class_col].unique(), classes)

    return percentage, all_equal, test_agent, train_agent

def make_dsads(path):
    dsads = scipy.io.loadmat(path)
    dsads_df = pd.DataFrame(dsads['data_dsads'])
    
    agents = dsads_df.iloc[:, 407].unique()
    test_agent = np.random.choice(agents, 2, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    # print(train_agent, test_agent)
    train_set, test_set = make(dsads_df, user_col=407, exclude=test_agent.tolist(), drop=405)
    # print(len(train_set["targets"]), len(test_set["targets"]))
    return train_set, test_set

def make_pamap(path):
    pamap = scipy.io.loadmat(path)
    pamap_df = pd.DataFrame(pamap['data_pamap'])

    percentage, all_equal, test_agent, train_agent = split_agents(pamap_df, 244, 243)

    while percentage < 0.20 or percentage > 0.26 or not all_equal:
        percentage, all_equal, test_agent, train_agent = split_agents(pamap_df, 244, 243)

    train_set, test_set = make(pamap_df, user_col=244, exclude=test_agent.tolist())
    return train_set, test_set

def make_hapt(path, totensor=False):
    train_x = np.loadtxt(f"{path}/Train/X_train.txt")
    train_y = np.loadtxt(f"{path}/Train/y_train.txt", dtype=np.int32)
    y_map = {v: k for k, v in enumerate(np.unique(train_y))}    
    if totensor:
        y = torch.tensor(np.vectorize(y_map.get)(train_y)).type(torch.int)
        data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(train_x, y)]
    else:
        y = np.vectorize(y_map.get)(train_y)
        data = [[x_, y_] for x_, y_ in zip(train_x, y)]
    train_data = {'data': data, 'targets': y}

    test_x = np.loadtxt(f"{path}/Test/X_test.txt")
    test_y = np.loadtxt(f"{path}/Test/y_test.txt", dtype=np.int32)
    if totensor:
        y = torch.tensor(np.vectorize(y_map.get)(test_y)).type(torch.int)
        data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(test_x, y)]
    else:
        y = np.vectorize(y_map.get)(test_y)
        data = [[x_, y_] for x_, y_ in zip(test_x, y)]
    test_data = {'data': data, 'targets': y}

    return train_data, test_data

def make_wisdm(path):
    # path should point to all.csv
    wisdm_df = pd.read_csv(path, header=None)
    agents = wisdm_df.iloc[:, 92].unique()
    test_agent = np.random.choice(agents, 10, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    train_set, test_set = make(wisdm_df, user_col=92, exclude=test_agent.tolist(), scale=True)
    return train_set, test_set 

def make_flexible4a(path, typ='dsads'):     
    if typ == 'dsads':
        new_col, final_col = 408, 405
        dsads = scipy.io.loadmat(path)
        dframe = pd.DataFrame(dsads['data_dsads'])
        dframe[new_col] = pd.Series(dtype=np.int32)
        class_ids = [i for i in range(1, 20)]
        user_ids = [i for i in range(1, 9)]
        user_col, cls_col = 407, 406
        to_drop = [405, 406, 407]        
    elif typ == 'wisdm':
        new_col, final_col = 93, 91
        dframe = pd.read_csv(path, header=None)
        dframe[new_col] = pd.Series(dtype=np.int32)
        class_ids = [i for i in range(18)]
        user_ids = [1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 
                    1613, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 
                    1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 
                    1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650]       
        user_col, cls_col = 92, 91
        to_drop = [91, 92]

    mesh_grid = np.transpose([np.tile(user_ids, len(class_ids)), np.repeat(class_ids, len(user_ids))])

    for i, m in enumerate(mesh_grid):
        # temp = dframe.loc[dframe[user_col] == m[0]]
        # if m[1] not in temp[cls_col].unique():
        #     print(f"User: {m[0]} does not have class: {m[1]}")
        dframe.loc[(dframe[cls_col] == m[1]) & (dframe[user_col] == m[0]), new_col] = i
    dframe[new_col] = dframe[new_col].astype(int)

    dframe.drop(dframe.columns[to_drop], axis=1, inplace=True)

    x_train, x_test = [], []
    for i in sorted(dframe[new_col].unique()):
        temp = dframe.loc[dframe[new_col] == i]
        x_tr, x_te = train_test_split(temp, test_size=0.2)
        x_train.append(x_tr)
        x_test.append(x_te)

    x_train = pd.concat(x_train)
    x_train.columns = [i for i in range(len(x_train.columns))]
    
    x_test = pd.concat(x_test)    
    x_test.columns = [i for i in range(len(x_test.columns))]

    # print(sorted(x_train[final_col].unique()), len(x_train[final_col].unique()))
    # print(sorted(x_test[final_col].unique()), len(x_test[final_col].unique()))
    # exit()

    if typ == 'wisdm':
        train_data = to_tensor(x_train, final_col, scale=True)
        test_data = to_tensor(x_test, final_col, scaler=train_data['scaler'])
        del train_data['scaler']
        return train_data, test_data

    return to_tensor(x_train, final_col), to_tensor(x_test, final_col)

def make_flexible4b(path, save_path):
    dsads = scipy.io.loadmat(path)
    dsads_df = pd.DataFrame(dsads['data_dsads'])

    class_ids = [i for i in range(19)]
    to_map = {int(v): k for k, v in enumerate(range(1, 20))}
    to_drop = [405, 407]    # drop 407, i.e. user id as split is done randomly        

    dsads_df[406] = dsads_df[406].map(to_map)
    dsads_df.drop(dsads_df.columns[to_drop], axis=1, inplace=True)

    prev = 0
    df_storage = {}
    x_train, x_test = [], []

    for c in class_ids:
        sub = dsads_df.loc[dsads_df[406] == c]
        sub = sub.drop([406], axis=1)
        sub = sub.reset_index(drop=True)
        
        kmeans = KMeans(n_clusters=5).fit(sub)
        pred = kmeans.predict(sub)
        
        for clust in sorted(np.unique(pred)):
            new_class = prev + clust
            temp = sub.iloc[np.where(pred == clust)[0]]
            temp = temp.reset_index(drop=True)  # otherwise there will be warnings

            cls_ = pd.Series([new_class for _ in range(temp.shape[0])]).values
            temp.loc[:, len(temp.columns)] = cls_

            x_tr, x_te = train_test_split(temp, test_size=0.2)
            x_train.append(x_tr)
            x_test.append(x_te)

            df_storage[new_class] = {"df": temp, "kmeans": kmeans, "full_df": sub}
        
        prev += len(np.unique(pred))

    x_train = pd.concat(x_train)
    x_test = pd.concat(x_test)

    pickle.dump(df_storage, open(save_path, "wb"))

    return to_tensor(x_train, 405), to_tensor(x_test, 405)

class BaseDataset(Dataset):
    def __init__(self, data):
        self.inputs = self._to_tensor(data['x'])
        self.labels = data['y']
        self.gate_labels = None
        if 'gate_label' in data:
            self.gate_labels = data['gate_label']
        self.is_feature = True  # always True for HAR

    def _to_tensor(self, x):
        """ convert x of np.ndarray to torch.Tensor """
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
        return torch.tensor(np.array(x))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        
        if self.gate_labels is not None:
            return x, y, self.gate_labels[index]
        
        return x, y

def get_data(
    train_path,
    test_path=None,
    val_path=None,
    num_tasks=5, 
    classes_in_first_task=None, 
    validation=0.2, 
    shuffle_classes=True, 
    k=2, 
    dummy=False,
    seed=None,
    kind='pamap',
    save_path=None,
    custom_class=None):
    """
    train_path: the path to .mat train file
    test_path: the path to .mat test file
    val_path: the path to .mat validation file
    num_tasks: the number of tasks, this may be ignored if classes_in_first_task is not None
    classes_in_first_task: the number of classes in the first task. If None, the classes are divided evenly per task
    validation: floating number for validation size (e.g. 0.20)
    shuffle_classes: True/False to shuffle the class order
    k: the number of classes in the remaining tasks (only used if classes_in_first_task is not None)
    dummy: set to True to get only a small amount of data (for small testing on CPU)
    kind: 'pamap', 'dsads' etc.
    custom_class: a list of #class/task. E.g. [3, 3, 4, 5, 4]

    return:
    data: a dictionary of dataset for each task
        data = {
            [{
                'name': task-0,
                'train': {
                    'x': [],
                    'y': []
                },
                'val': {
                    'x': [],
                    'y': []
                },
                'test': {
                    'x': [],
                    'y': []
                },
                'classes': int
            }],
        }
    class_order: the order of the classes in the dataset (may be shuffled)
    """

    data = {}
    taskcla = []

    # trainset = torch_dataset(root=data_path, train=True, download=True, transform=transform)
    # testset = torch_dataset(root=data_path, train=False, download=True, transform=transform)

    if kind == 'dsads':
        trainset, testset = make_dsads(train_path)
        classes_in_first_task = 3
        num_tasks = 9
        k = 2
    elif kind == 'pamap':
        trainset, testset = make_pamap(train_path)
        num_tasks = 6
    elif kind == 'hapt':
        num_tasks = 6
        trainset, testset = make_hapt(train_path, totensor=True)
    elif kind == 'wisdm':
        num_tasks = 9
        trainset, testset = make_wisdm(train_path)
    elif kind == 'flex':
        trainset, testset = make_flexible4a(train_path, typ='dsads')
        classes_in_first_task = 12
        k = 10
    elif kind == 'flex2':
        trainset, testset = make_flexible4b(train_path, save_path)
        classes_in_first_task = 14
        k = 10
    elif kind == 'wisdmflex':
        trainset, testset = make_flexible4a(train_path, typ='wisdm')
        classes_in_first_task = 34
        k = 20        

    num_classes = len(np.unique(trainset["targets"]))
    class_order = list(range(num_classes))    
    
    if shuffle_classes:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(class_order)
    print("CLASS_ORDER:", class_order)

    if custom_class is not None:
        cpertask = np.array(custom_class)
        num_tasks = len(cpertask)
    elif classes_in_first_task is None:
        # Divide evenly the number of classes for each task
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        # Allocate the rest of the classes based on k
        remaining_classes = num_classes - classes_in_first_task
        cresttask = remaining_classes // k
        cpertask = np.array([classes_in_first_task] + [remaining_classes // cresttask] * cresttask)
        for i in range(remaining_classes % k):
            cpertask[i + 1] += 1
        num_tasks = len(cpertask)    

    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    total_task = num_tasks
    for tt in range(total_task):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        # data[tt]['nclass'] = cpertask[tt]

    # Populate the train set
    for i, (this_input, this_label) in enumerate(trainset["data"]):
        this_label = int(this_label)
        mapped_label = class_order.index(this_label)
        this_task = (mapped_label >= cpertask_cumsum).sum()

        data[this_task]['trn']['x'].append(this_input)
        data[this_task]['trn']['y'].append(mapped_label)
        # data[this_task]['trn']['y'].append(mapped_label - init_class[this_task])

        if dummy and i >= 500:
            break

    # Populate the test set
    for i, (this_input, this_label) in enumerate(testset["data"]):
        this_label = int(this_label)
        mapped_label = class_order.index(this_label)
        this_task = (mapped_label >= cpertask_cumsum).sum()

        data[this_task]['tst']['x'].append(this_input)        
        data[this_task]['tst']['y'].append(mapped_label)
        # data[this_task]['tst']['y'].append(mapped_label - init_class[this_task])

        if dummy and i >= 100:
            break

    # Populate validation if required
    if validation > 0.0:
        for tt in data.keys():
            pop_idx = [i for i in range(len(data[tt]["trn"]["x"]))]
            val_idx = random.sample(pop_idx, int(np.round(len(pop_idx) * validation)))
            val_idx.sort(reverse=True)

            for ii in range(len(val_idx)):
                data[tt]['val']['x'].append(data[tt]['trn']['x'][val_idx[ii]])
                data[tt]['val']['y'].append(data[tt]['trn']['y'][val_idx[ii]])
                data[tt]['trn']['x'].pop(val_idx[ii])
                data[tt]['trn']['y'].pop(val_idx[ii])

    for tt in range(total_task):
        data[tt]["classes"] = np.unique(data[tt]["trn"]["y"])
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        # print(f"data[{tt}]['classes']: {data[tt]['classes']}")
        # print(f"data[{tt}]['ncla']: {data[tt]['ncla']}")

    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    class_group = list(grouper(class_order, cpertask[0], cpertask=cpertask[1])) if classes_in_first_task is None else list(grouper(class_order, cpertask.tolist()))
    print(f"class_group: {class_group}")
    ordered = {class_order.index(c): k for k in range(len(class_group)) for c in class_group[k]}
    
    for tt in range(total_task):
        # better to have a uniform numpy, i.e. transform the tensor in wisdm to numpy
        # if kind == 'wisdm':
        data[tt]['trn']['x'] = torch.stack(data[tt]['trn']['x']).numpy().astype(np.float32)
        data[tt]['tst']['x'] = torch.stack(data[tt]['tst']['x']).numpy().astype(np.float32)
        data[tt]['val']['x'] = torch.stack(data[tt]['val']['x']).numpy().astype(np.float32)       

        data[tt]['trn']['y'] = np.array(data[tt]['trn']['y'], dtype=np.int32)
        data[tt]['tst']['y'] = np.array(data[tt]['tst']['y'], dtype=np.int32)
        data[tt]['val']['y'] = np.array(data[tt]['val']['y'], dtype=np.int32)
        # else:
        #     data[tt]['trn']['x'] = np.array(data[tt]['trn']['x'], dtype=np.float32)
        #     data[tt]['tst']['x'] = np.array(data[tt]['tst']['x'], dtype=np.float32)
        #     data[tt]['val']['x'] = np.array(data[tt]['val']['x'], dtype=np.float32)        

        #     data[tt]['trn']['y'] = np.array(data[tt]['trn']['y'], dtype=np.int32)
        #     data[tt]['tst']['y'] = np.array(data[tt]['tst']['y'], dtype=np.int32)
        #     data[tt]['val']['y'] = np.array(data[tt]['val']['y'], dtype=np.int32)

        data[tt]['ordered'] = ordered

    return data, taskcla, class_order

def grouper(iterable, n, cpertask=2, fillvalue=None):
    if isinstance(n, list):
        i = 0
        group = []
        for cp in n:
            group.append(tuple(iterable[i:i+cp]))
            i += cp
        return group
    else:
        group = [tuple(iterable[:n])]
        remaining = len(iterable) - n
        for i in range(remaining // cpertask):
            start = n + (cpertask * i)
            group.append(tuple(iterable[start:start+2]))
        return group     

if __name__ == '__main__':
    path = ['/Users/fahrurrozirahman/Documents/elquinto/har/DSADS/pamap.mat', '/Users/fahrurrozirahman/Documents/elquinto/har/DSADS/dsads.mat']
    kind = ['pamap', 'dsads']
    d = 0
    data, taskcla, class_order = get_data(path[d], kind=kind[d])