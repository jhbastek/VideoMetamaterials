import torch
import torch.nn.functional as F
import numpy as np

class Normalization:
    def __init__(self,data,dataType,strategy):
        self.mu = torch.mean(data,dim=0)
        self.std = torch.std(data,dim=0)
        self.min = torch.min(data,dim=0)[0]
        self.max = torch.max(data,dim=0)[0]
        self.globalmin = torch.min(data)
        self.globalmax = torch.max(data)
        self.dataType = dataType
        self.cols = data.size()[1]
        self.strategy = strategy
    
    def normalize(self, data):
        list_index_cat = []       
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            if self.dataType[i] == 'continuous':

                if(self.strategy == 'min-max-1'):
                    #scale to [0,1]
                    temp[:,i] = torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])

                elif(self.strategy == 'global-min-max-1'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)

                elif(self.strategy == 'min-max-2'):
                    #scale to [-1,1]
                    temp[:,i] = 2.*torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])-1.

                elif(self.strategy == 'global-min-max-2'):
                    #scale to [-1,1] based on min max of full dataset
                    temp[:,i] = 2.*torch.div(data[:,i]-self.globalmin, self.globalmax-self.globalmin)-1.

                elif(self.strategy == 'mean-std'):
                    #scale s.t. mean=0, std=1
                    temp[:,i] = torch.div(data[:,i]-self.mu[i], self.std[i])

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')

            elif self.dataType[i] == 'categorical':
                #convert categorical features into binaries and append at the end of feature tensor
                temp = torch.cat((temp,F.one_hot(data[:,i].to(torch.int64))),dim=1)
                list_index_cat = np.append(list_index_cat,i)
                                   
            else:
                raise ValueError("Data type must be either continuous or categorical")

        # delete original (not one-hot encoded) categorical features
        j = 0
        for i in np.array(list_index_cat, dtype=np.int64):          
            temp = torch.cat([temp[:,0:i+j], temp[:,i+1+j:]],dim=1)
            j -= 1

        return temp

    def unnormalize(self, data):
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            if self.dataType[i] == 'continuous':
                
                if(self.strategy == 'min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-1'):
                    temp[:,i] = torch.mul(data[:,i], self.globalmax-self.globalmin) +self.globalmin

                elif(self.strategy == 'min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.max[i]-self.min[i]) +self.min[i]

                elif(self.strategy == 'global-min-max-2'):
                    temp[:,i] = torch.mul(0.5*data[:,i]+0.5, self.globalmax-self.globalmin) +self.globalmin
            
                elif(self.strategy == 'mean-std'):
                    temp[:,i] = torch.mul(data[:,i], self.std[i]) + self.mu[i]

                elif (self.strategy == 'none'):
                    temp[:,i] = data[:,i]

                else:
                    raise ValueError('Incorrect normalization strategy')
                
            elif self.dataType[i] == 'categorical':
                temp[:,i] = data[:,i]

            else:
                raise ValueError("Data type must be either continuous or categorical")
        return temp