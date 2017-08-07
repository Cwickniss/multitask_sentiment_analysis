import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils import nb_classes
from utils import max_sentence_size
from utils import batch_generator

# The size of the character embedding used in all
# other tasks.
embedding_size = 50

class CharacterLanguageModel(nn.Module):
    """ The Word-Character Level Embedding Module. 
    """
    def __init__(self):
        super(CharacterLanguageModel, self).__init__()

        self.embedding = nn.Embedding(nb_classes, embedding_size)

    def forward(self, x):
        arr = list()

        for sentence in x:
            # Sentence embedding
            sent_emb = list()

            for word in sentence:
                word = np.array(word)
                word = torch.from_numpy(word)
                word = Variable(word)

                # Gets the embedding for each character in
                # the word
                char_emb = self.embedding(word)

                # Computes the mean between all character level
                # embeddings. MxN -> 1xN
                char_emb = torch.mean(char_emb, 0)

                sent_emb.append(char_emb)

            arr.append(sent_emb)

        return arr