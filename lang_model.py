import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils import nb_classes
from utils import max_sentence_size

embedding_size = 50

class CharacterLanguageModel(nn.Module):
    def __init__(self):
        super(CharacterLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(nb_classes, embedding_size)
        
    def forward(self, x):
        arr = torch.zeros(len(x), max_sentence_size, embedding_size)
        
        for i, bow in enumerate(x):
            j = 0
            
            for vec in bow:
                vec = np.unique(vec)
                vec = torch.from_numpy(vec).long()
                vec = Variable(vec)
                
                emb = self.embedding(vec)
                
                # Squeezes the embedding taking the average of each 
                # component. MxN -> 1xN
                emb = torch.mean(emb, 0)
                
                arr[i][j] = emb.data
                j += 1

        return arr
