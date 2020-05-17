import pandas as pd
import numpy as np

class FinancialDataLoader():
    def __init__(self, data):
        self.data = data
        self.n_vars = data.shape[1]
        self.data_to_use = None
    
    def rewind(self, argument):
        ## Some reset here
        self.data_to_use = None
    
    def get_batch(self, batchsize, songlength, random_offset=True, random_selection=True):
        ## It's not a song, but if I do it this way I don't have to modify the code of the CRNNGAN class.
        offset = 0
        if random_offset:
            offset = np.random.randint(0, songlength)
        total_seqs = int((len(self.data)-offset)/songlength)
        if total_seqs < batchsize:
            print('The batch size or the sequence length are too large')
            return (None, None)
        if self.data_to_use is None:
            self.data_to_use = [
                self.data.iloc[i*songlength+offset:(i+1)*songlength+offset].values.tolist()
                for i in range(total_seqs)
            ]
        indexes = [i for i in range(batchsize)]
        '''
        if random_selection:
            indexes = random.sample([i for i in range(len(self.data_to_use))], k=batchsize)
        '''
        if batchsize > len(self.data_to_use):
            self.data_to_use = []
            return (None, None)
        data_to_return = []
        for i in range(batchsize):
            index = np.random.randint(len(self.data_to_use)) if random_selection else indexes[i]
            data_to_return.append(self.data_to_use.pop(index))
        data_to_return = np.asarray(data_to_return).reshape(batchsize, songlength, self.n_vars)
        return (None, data_to_return)