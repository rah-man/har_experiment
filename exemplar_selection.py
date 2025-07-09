import numpy as np
import random
import torch

class ExemplarSelector:
    def __init__(self, total_exemplar=0, exemplar_per_class=0):
        self.buffer_x = []
        self.buffer_y = []
        self.total_exemplar = total_exemplar
        self.exemplar_per_class = exemplar_per_class

    def pool_buffer(self, x, y):
        pool_x, pool_y = [], []
        pool_x.extend(self.buffer_x)
        pool_y.extend(self.buffer_y)
        
        pool_x.extend(x)
        pool_y.extend(y)

        return pool_x, pool_y

    def exemplars_per_class_num(self, num_cls):
        if self.exemplar_per_class:
            return self.exemplar_per_class

        num_exemplars = self.total_exemplar
        exemplar_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplar_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplar_per_class

    def update_buffer(self, x, y, num_cls):
        pass

class HerdingExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, total_exemplar=0, exemplar_per_class=0):
        super().__init__(total_exemplar, exemplar_per_class)

    def update_buffer(self, x, y, num_cls):
        result = []

        pool_x, pool_y = self.pool_buffer(x, y)
        feats = torch.tensor(np.array(pool_x))
        exemplar_per_class = self.exemplars_per_class_num(num_cls)

        for curr_cls in range(num_cls):
            cls_ind = np.where(np.array(pool_y) == curr_cls)[0]     
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)

            cur_exemplar_per_class = exemplar_per_class
            if exemplar_per_class > len(cls_ind):
                print(f"Not enough samples to store. Select all samples instead.\tNeeded: {exemplar_per_class}")
                cur_exemplar_per_class = len(cls_ind)

            cls_feats = feats[cls_ind]
            cls_mu = cls_feats.mean(0)
            selected = []
            selected_feat = []            
            for k in range(cur_exemplar_per_class):
                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf
                for item in cls_ind:
                    if item not in selected:
                        feat = feats[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat)
                selected.append(newone)
            result.extend(selected)        
        
        self.buffer_x = [pool_x[i] for i in result]
        self.buffer_y = [pool_y[i] for i in result]        


class RandomExemplarsSelector(ExemplarSelector):
    def __init__(self, total_exemplar=0, exemplar_per_class=0):
        super().__init__(total_exemplar, exemplar_per_class)

    def update_buffer(self, x, y, num_cls):
        result = []
        pool_x, pool_y = self.pool_buffer(x, y)
        exemplar_per_class = self.exemplars_per_class_num(num_cls)
        for curr_cls in range(num_cls):
            cls_ind = np.where(np.array(pool_y) == curr_cls)[0]        
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)

            cur_exemplar_per_class = exemplar_per_class
            if exemplar_per_class > len(cls_ind):
                print(f"Not enough samples to store. Select all samples instead.\tNeeded: {exemplar_per_class}")
                cur_exemplar_per_class = len(cls_ind)

            result.extend(random.sample(list(cls_ind), cur_exemplar_per_class))
        
        self.buffer_x = [pool_x[i] for i in result]
        self.buffer_y = [pool_y[i] for i in result]