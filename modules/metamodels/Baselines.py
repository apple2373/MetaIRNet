import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class Baselines(nn.Module):
    '''
    Simple  baselines
    In meta-train time, this will do normal supervised learning
    In meta-test time, this will do softmax-regression/logistic-regression/nearest-neighbor 
    '''
    def __init__(self,method,feature_dim,num_train_classes):
        '''
        method: method name one of  {softmax,logistic,nearest} corresponding to softmax-regression/logistic-regression/nearest-neighbor 
        feature_dim: dimension of embeddings (or feaure space)
        num_train_classes: number of training classes for supervised training
        '''
        super(Baselines, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_train_classes)
        self.method = method
        
    def forward(self,support_embs, support_labels=None, query_embs=None):
        if self.training:
            scores = self.fc(support_embs)
            return scores
        else:
            support = support_embs.detach().cpu().numpy()
            query   =   query_embs.detach().cpu().numpy()

            y_support = support_labels.detach().cpu().numpy()
            
            if self.method == "softmax":
                clf = LogisticRegression(solver='lbfgs',multi_class='multinomial')
            elif self.method == "logistic":
                clf = LogisticRegression(solver='lbfgs',multi_class='ovr')
            elif self.method == "nearest":
                clf = KNeighborsClassifier(n_neighbors=1)
            else:
                raise NotImplementedError("this method is not defined",self.method)
                
            clf.fit(support,y_support)
                        
        return torch.from_numpy(clf.predict_proba(query)).to(support_embs.device)