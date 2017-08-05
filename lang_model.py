import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils import nb_classes
from utils import max_sentence_size
from utils import batch_generator

embedding_size = 50

class CharacterLanguageModel(nn.Module):
    def __init__(self):
        super(CharacterLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(nb_classes, embedding_size)

    def forward(self, x):
        arr = list()

        for sentence in x:
            sent_emb = list()
            
            for word in sentence:
                emb = np.array(word)
                emb = torch.from_numpy(emb)
                emb = Variable(emb)
                
                emb = self.embedding(emb)
                emb = torch.mean(emb, 0)
                
                sent_emb.append(emb)
            
            arr.append(sent_emb)

        return arr