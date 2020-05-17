import torch
from torch import nn, optim
import numpy as np
import time
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels=4, hidden_dim=350, n_layers=3, scale=False):
        super().__init__()
        self.scale = scale
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        ## in_channels represents the number of features 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = nn.LSTM(
            input_size = self.in_channels, hidden_size = self.hidden_dim,
            num_layers = self.n_layers, batch_first = True
        )
        # We want the output to be a sequence of seq_length in which each value in the sequence
        # has 4 channels. Therefore, we'll need a tensor of size batch_size*seq_length*num_channels
        # For that, we need a linear layer that receives a tensor of size (batch_size, seq_length, hidden_dim)
        # and returns a tensor of size (batch_size, seq_length, out_channels)
        if self.scale:
            #self.sigmoid = nn.Sigmoid()
            self.relu6 = nn.ReLU6()
        self.fc = nn.Linear(in_features = self.hidden_dim, out_features = self.out_channels)
        
    def forward(self, x, h=None):
        if (h==None):
            lstm_output, hidden = self.lstm(x)
        else:
            lstm_output, hidden = self.lstm(x, h)
        # Stack up LSTM outputs
        out = lstm_output.reshape(-1, self.hidden_dim) 
        out = self.fc(out)
        if self.scale == 'minmax':
            #out = self.sigmoid(out)
            out = self.relu6(out)*(1.0/6)
        return out, hidden

class ComplexGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=4, hidden_dim=350, n_layers=3, scale=False):
        super().__init__()
        self.scale = scale
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        ## in_channels represents the number of features 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = nn.LSTM(
            input_size = self.in_channels, hidden_size = self.hidden_dim,
            num_layers = self.n_layers, batch_first = True
        )
        # We want the output to be a sequence of seq_length in which each value in the sequence
        # has 4 channels. Therefore, we'll need a tensor of size batch_size*seq_length*num_channels
        # For that, we need a linear layer that receives a tensor of size (batch_size, seq_length, hidden_dim)
        # and returns a tensor of size (batch_size, seq_length, out_channels)
        if self.scale:
            #self.sigmoid = nn.Sigmoid()
            self.relu6 = nn.ReLU6()
        self.fc_1 = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(in_features = self.hidden_dim, out_features = self.out_channels)
        
    def forward(self, x, h=None):
        if (h==None):
            lstm_output, hidden = self.lstm(x)
        else:
            lstm_output, hidden = self.lstm(x, h)
        # Stack up LSTM outputs
        out = lstm_output.reshape(-1, self.hidden_dim) 
        out = self.relu(self.fc_1(out))
        out = self.fc_out(out.reshape(-1, self.hidden_dim))
        if self.scale == 'minmax':
            #out = self.sigmoid(out)
            out = self.relu6(out)*(1.0/6)
        return out, hidden

class Discriminator(nn.Module):
    def __init__(self, out_size=2, in_channels=4, hidden_dim=350, n_layers=2, bidirectional=False):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_size = out_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size = self.in_channels, hidden_size = self.hidden_dim,
            num_layers = self.n_layers, batch_first = True, bidirectional = self.bidirectional
        )
        if self.bidirectional:
            self.fc = nn.Linear(in_features = self.hidden_dim*2, out_features = 2)
        else:
            self.fc = nn.Linear(in_features = self.hidden_dim, out_features = 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, seq_length, h=None):
        # I could just get seq_length out of x's shape, but it's easier to pass
        # it as an argument to the function.
        if (h==None):
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, h)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        if self.bidirectional:
            out = out.reshape(-1, seq_length, self.hidden_dim*2)
        else:
            out = out.reshape(-1, seq_length, self.hidden_dim)
        out = self.logsoftmax(self.fc(out))
        
        # return the final output and the hidden state
        return out, hidden

class ComplexDiscriminator(nn.Module):
    def __init__(self, out_size=2, in_channels=4, hidden_dim=350, n_layers=2, bidirectional=False):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_size = out_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size = self.in_channels, hidden_size = self.hidden_dim,
            num_layers = self.n_layers, batch_first = True, bidirectional = self.bidirectional
        )
        if self.bidirectional:
            self.fc_1 = nn.Linear(in_features = self.hidden_dim*2, out_features = self.hidden_dim*2)
            self.fc_out = nn.Linear(in_features = self.hidden_dim*2, out_features = 2)
        else:
            self.fc_1 = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim)
            self.fc_out = nn.Linear(in_features = self.hidden_dim, out_features = 2)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, seq_length, h=None):
        if h is not None:
            out, hidden = self.lstm(x, h)
        else:
            out, hidden = self.lstm(x)
        if self.bidirectional:
            out = out.reshape(-1, seq_length, self.hidden_dim*2)
        else:
            out = out.reshape(-1, seq_length, self.hidden_dim)
        out = self.relu(self.fc_1(out))
        out = self.logsoftmax(self.fc_out(out))
        
        return out, hidden

def welschTTest(input, target, reduction='mean'):
    ### Not done here, I should be computing the p-values!!!!
    #print('input: ', input)
    num_features = input.size()[-1]
    input = input.view(-1, num_features)
    target = target.view(-1, num_features)
    size_input = input.size()[0]
    size_target = target.size()[0]
    std_input, mean_input = torch.std_mean(input, dim=0, keepdim=True)
    std_target, mean_target = torch.std_mean(target, dim=0, keepdim=True)
    var_input = torch.pow(std_input, 2)
    var_target = torch.pow(std_target, 2)
    denominator = torch.pow(
        torch.add(
            torch.div(var_input, size_input),
            torch.div(var_target, size_target)
        ),
        0.5
    )
    numerator = torch.add(mean_input, other=mean_target, alpha=-1)

    v_numerator = torch.pow(denominator, 4)
    v_denominator = torch.add(
        torch.div(
            torch.pow(var_input, 2),
            torch.mul(torch.pow(torch.tensor(size_input), 2), size_input - 1)
        ),
        other = torch.div(
            torch.pow(var_target, 2),
            torch.mul(torch.pow(torch.tensor(size_target), 2), size_target - 1)
        )
    )

    t_statistic_all_vars = torch.div(numerator, denominator)
    v_degrees_all_vars = torch.div(v_numerator, v_denominator)

    p_vals = []
    for t_statistic, v_degrees in zip(t_statistic_all_vars[0], v_degrees_all_vars[0]):
        distr = torch.distributions.studentT.StudentT(df=v_degrees)
        p_vals.append(torch.exp(distr.log_prob(t_statistic)))
    
    p_vals = torch.stack(p_vals)
    print(p_vals)
    
    
    if reduction == 'mean':
        p_vals_out = torch.mean(p_vals)
    elif reduction == 'median':
        p_vals_out = torch.median(p_vals)
    elif reduction == 'max':
        p_vals_out = torch.max(p_vals)
        
    return t_statistic


class CRNNGAN():
    def __init__(self, batch_length, sequence_length,
                 in_channels_g, out_channels_g, hidden_dim_g, n_layers_g,
                 in_channels_d, out_channels_d, hidden_dim_d, n_layers_d,
                 epochs = 15, lr_g = 0.1, lr_d = 0.1, beta_g = 0.5, beta_d = 0.5,
                 curriculum_learning = False, seq_length_increments = 5,
                 G_var_threshold = 0.01, D_var_threshold = 0.01,
                 max_sequence_length = 100, print_every=10,
                 scale=None, scale_values=None, linear_layers=1,
                 complexGenerator = False, complexDiscriminator=False,
                 bidirectional=False, ttest=False, reg_ttest=0.1):
        self.batch_length = batch_length
        self.sequence_length = sequence_length
        self.in_channels_g = in_channels_g
        self.out_channels_g = out_channels_g
        self.hidden_dim_g = hidden_dim_g
        self.n_layers_g = n_layers_g
        self.in_channels_d = in_channels_d
        self.out_channels_d = out_channels_d
        self.hidden_dim_d = hidden_dim_d
        self.n_layers_d = n_layers_d
        
        self.epochs = epochs
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta_g = beta_g
        self.beta_d = beta_d
        
        self.curriculum_learning = curriculum_learning
        self.seq_length_increments = seq_length_increments
        self.max_sequence_length = max_sequence_length
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.print_every = print_every
        self.scale = scale
        self.scale_values = scale_values
        
        self.complexGenerator = complexGenerator
        self.complexDiscriminator =  complexDiscriminator
        self.bidirectional = bidirectional

        if self.complexGenerator:
            print('Complex Generator')
            self.netG = ComplexGenerator(
                in_channels=self.in_channels_g, out_channels=self.out_channels_g,
                hidden_dim=self.hidden_dim_g, n_layers=self.n_layers_g,
                scale=self.scale
            )
            self.netG.to(self.device)
        else:
            self.netG = Generator(
                in_channels=self.in_channels_g, out_channels=self.out_channels_g,
                hidden_dim=self.hidden_dim_g, n_layers=self.n_layers_g,
                scale=self.scale
            )
            self.netG.to(self.device)
        
        if self.complexDiscriminator:
            print('Complex Discriminator')
            self.netD = ComplexDiscriminator(
                in_channels=self.in_channels_d, out_size=self.out_channels_d,
                hidden_dim=self.hidden_dim_d, n_layers=self.n_layers_d, 
                bidirectional=self.bidirectional
            )
            self.netD.to(self.device)
        else:
            self.netD = Discriminator(
                in_channels=self.in_channels_d, out_size=self.out_channels_d,
                hidden_dim=self.hidden_dim_d, n_layers=self.n_layers_d,
                bidirectional=self.bidirectional
            )
            self.netD.to(self.device)
        

        self.ttest = ttest
        self.reg_ttest = reg_ttest
        self.criterion = nn.NLLLoss()
        
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(self.beta_g, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(self.beta_d, 0.999))
        
        self.G_losses = []
        self.D_losses = []
        self.G_losses_variance = []
        self.G_var_threshold = G_var_threshold
        self.D_losses_varaince = []
        self.D_var_threshold = D_var_threshold
        self.generated_songs = []
        self.fixed_noises = [torch.randn(self.batch_length, self.in_channels_g, self.sequence_length, 1, 1).to(self.device)]

    def scaler(self, batch_data):
        new_batch = []
        if self.scale == 'minmax':
            # batch_data.shape[2] has the number of features
            for i in range(batch_data.shape[2]):
                min_i, max_i = self.scale_values[i]
                new_batch.append((batch_data[:,:,i] - min_i)/(max_i-min_i))
            return np.clip(np.asarray(new_batch).reshape(batch_data.shape[0], batch_data.shape[1], -1), 0, 1)
        elif self.scale == 'standard':
            for i in range(batch_data.shape[2]):
                mean_i, std_i = self.scale_values[i]
                new_batch.append((batch_data[:,:,i] - mean_i)/std_i)
            return np.asarray(new_batch).reshape(batch_data.shape[0], batch_data.shape[1], -1)
        
        
    def trainloop(self, dataLoader):
        print('Start training for {} epochs'.format(self.epochs))
        iters = 0
        iters_per_epoch = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            ## Reset data loader to load new batches
            dataLoader.rewind('train')
            ## We only care about the four variables obtain from the midi file (we can scrap the metadata)
            _, batch_data = dataLoader.get_batch(batchsize=self.batch_length, songlength=self.sequence_length)
            ## Since there's no "easy" way of obtaining the number of batches, we'll loop until there are no
            ## more batches, i.e. a batch is None
            while batch_data is not None:
                if self.scale is not None:
                    batch_data_scaled = self.scaler(batch_data)
                else:
                    batch_data_scaled = batch_data
                batch_data_scaled = torch.FloatTensor(batch_data_scaled).to(self.device)
                ### First we train the Discriminator Network
                ## Real Data
                self.netD.zero_grad()
                # Construct a (batch_length, 2) long tensor with all zeroes in the first column
                # and all ones in the second one
                label_real = torch.full((self.batch_length,2), 1).long()
                label_real[:,0] = 0
                label_real = label_real.to(self.device)
                
                # Here the data does not come as a tensor and therefore we have to convert it
                output_real, _ = self.netD.forward(
                    batch_data_scaled, 
                    seq_length=self.sequence_length
                )
                loss_real = self.criterion(output_real, label_real)
                loss_real.backward()
                # Since the class according to the model is the one with a larger value,
                # we'll take the mean of the max on a row basis. (The output has a shape
                # which is (batch_lenght, 2), and the class (fake/real) is determined as
                # the larger of the two columns).
                D_x = torch.max(output_real, dim=1)[0].mean().item()
                
                ## Fake Data
                # Generate noise (self.in_channels_g determines how the noise is to be created)
                noise = torch.randn(self.batch_length, self.in_channels_g, self.sequence_length, 1, 1)
                noise = noise.to(self.device)
                # label_fake is just label_real with the columns swapped
                label_fake = torch.index_select(label_real, 1, torch.LongTensor([1, 0]).to(self.device))
                ### NOT SURE IF THIS ONE IS NEEDED
                label_fake = label_fake.to(self.device)
                # Get the fake sequence out of the generator
                fake_data, _ = self.netG.forward(noise.reshape(-1, 1, self.in_channels_g))
                fake_data = fake_data.to(self.device)
                # Get the output for the discriminator
                output_fake, _ = self.netD.forward(
                    fake_data.reshape(self.batch_length, self.sequence_length, -1),
                    self.sequence_length
                )
                loss_fake = self.criterion(output_fake, label_fake)
                # Here we call retain graph so that it isn't lost after calling the optimizer
                # for the discriminator, as we will need part of the graph to be used again
                # for the generator's optimizer.
                loss_fake.backward(retain_graph=True)
                D_G_z1 = torch.max(output_fake, dim=1)[0].mean().item()
                # Add gradients from real and fake batches
                loss_discriminator = torch.add(loss_real, loss_fake)
                # loss_discriminator = loss_real + loss_fake
                self.optimizerD.step()
                
                ### Now, we train the Generator Network
                self.netG.zero_grad()
                # From the generator's cost point of view fake labels are real labels and vice-versa
                label_fake = label_real
                
                # Output of the discriminator network with fake data
                output_fake, _ = self.netD.forward(
                    fake_data.reshape(self.batch_length, self.sequence_length, -1),
                    self.sequence_length
                )
                loss_fake_G = self.criterion(output_fake, label_fake)
                if self.ttest:
                    loss_fake_G = torch.add(
                        self.criterion(output_fake, label_fake), 
                        welschTTest(batch_data_scaled, fake_data.reshape(self.batch_length, self.sequence_length, -1)), 
                        alpha=self.reg_ttest
                    )
                loss_fake_G.backward()
                D_G_z2 = torch.max(output_fake, dim=1)[0].mean().item()
                self.optimizerG.step()
                
                if iters % int(self.print_every) == 0:
                    print('[%d/%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\n Time since epoch started: %.2f'
                          % (epoch, self.epochs, iters,
                             loss_discriminator.item(), loss_fake_G.item(), 
                             D_x, D_G_z1, D_G_z2, 
                             time.time()-start_time))
                    print('There are a total of {} fake songs saved already'.format(len(self.generated_songs)))
            
                self.G_losses.append(loss_fake_G.item())
                self.D_losses.append(loss_discriminator.item())
                count = 10 if iters > 10 else iters
                if self.curriculum_learning:
                    self.G_losses_variance.append(np.var(self.G_losses[-count:]))
                    self.D_losses_varaince.append(np.var(self.D_losses[-count:]))
            
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 2 == 0) or ((epoch == self.epochs-1)):
                    with torch.no_grad():
                        fake, _ = self.netG.forward(self.fixed_noises[-1].reshape(-1, 1, self.in_channels_g))
                        self.generated_songs.append(fake.reshape(self.batch_length, self.sequence_length, -1))
                
                iters += 1
                iters_per_epoch += 1
                _, batch_data = dataLoader.get_batch(batchsize=self.batch_length, songlength=self.sequence_length)
            if self.curriculum_learning and (
                np.mean(self.G_losses_variance[-iters_per_epoch:]) < self.G_var_threshold and
                np.mean(self.D_losses_varaince[-iters_per_epoch:]) < self.D_var_threshold
            ):
            ## Instead of the mean of the variances I could just update the sequence length if 70% of the variances
            ## are below a certain threshold
            # if (self.curriculum_learning and (
            #    sum(np.asarray(self.G_losses_variance[-iters_per_epoch:]) < self.G_var_threshold)/iters_per_epoch >= 0.7 and
            #    sum(np.asarray(self.D_losses_varaince[-iters_per_epoch:]) < self.D_var_threshold)/iters_per_epoch >= 0.7
            # )):
                if (self.sequence_length < self.max_sequence_length):
                    self.sequence_length += self.seq_length_increments
                    print('New sequence length: ', self.sequence_length, ' modified on iteration ', iters)
                    print('The variances were {:.5f} for G and {:.5f} for D'.format(self.G_losses_variance[-1], self.D_losses_varaince[-1]))
                    self.fixed_noises.append(torch.randn(self.batch_length, self.in_channels_g, self.sequence_length, 1, 1).to(self.device))
                else:
                    print('Maximum sequence reached...')
            iters_per_epoch = 0
            print('Epoch time: {:.2f}'.format(time.time()-start_time))

    def plot_losses(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

# Auxiliary functions to ease out the comprehension of the notebook.
def min_max(dataLoader, n_iters=10):
    """
    This function computes the minimum and the maximum given the data loader being used.
    This data loader has been borrowed from Olof Mogren's job, and it allows the loading
    of random batches of data of a given batch length and sequence length.
    Since there's some randomness in the process, I'll be computiong the minimum and maximum
    for a number of iterations specified by intput parameter 'n_iter'.
    """
    min_batch = []
    max_batch = []
    batch_size = 200
    ## We'll keep this number similar to what we normally get.
    seq_length = 200
    ## Since we get random batches, we'll run this for some epochs.
    for _ in range(n_iters):
        dataLoader.rewind('train')
        _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')
        while batch_data is not None:
            min_batch.append([batch_data.min(axis=0)])
            max_batch.append([batch_data.max(axis=0)])
            _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')    
    min_batch = np.array(min_batch)
    min_total = min_batch.reshape(-1, 4).min(axis=0)
    max_batch = np.array(max_batch)
    max_total = max_batch.reshape(-1, 4).max(axis=0)  
    return min_total, max_total

def get_percentile(dataLoader, percentile, n_iters=50):
    """
    This function allows for the computation of an approximation to the percentile of the 
    four variables provided by the data loader in a batch basis. Since computing the whole
    data set while loading the data in samples is not possible we'll approximate these values
    by computing the percentile for each batch, average it through all the bathces in the whole
    dataset and repeat this 'n_iters' times.
    """
    batch_size = 200
    seq_length = 200
    percentiles_out = []
    for i in range(n_iters):
        percentiles = []
        dataLoader.rewind('train')
        _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')
        while batch_data is not None:
            percentiles.append(np.percentile(batch_data.reshape(-1, 4), percentile, axis=0))
            _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')

        percentiles = np.array(percentiles)
        perc_total = percentiles.reshape(-1, 4).mean(axis=0)
        percentiles_out.append(perc_total)
    percentiles_out = np.array(percentiles_out)
    perc_out_total = percentiles_out.mean(axis=0)
    return perc_out_total

def compute_mean_and_std(dataLoader, n_iters=10):
    """
    This function computes an approximation of the mean and standard deviation in a similar manner
    to how the 'get_percentile' function does it.
    """
    batch_size = 200
    seq_length = 200
    means_out = []
    stds_out = []
    for i in range(n_iters):
        means = []
        stds = []
        dataLoader.rewind('train')
        _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')
        while batch_data is not None:
            means.append(batch_data.reshape(-1, 4).mean(axis = 0))  
            stds.append(batch_data.reshape(-1, 4).std(axis=0))
            _, batch_data = dataLoader.get_batch(batch_size, seq_length, 'train')
        means_out.append(np.asarray(means).mean(axis=0))
        stds_out.append(np.asarray(stds).mean(axis=0))
    return np.asarray(means_out).mean(axis=0), np.asarray(stds_out).mean(axis=0)

def unscale(song, min_total, max_total):
    """
    This function takes as input a song generated by the Generator Network trianed with 
    scaled data and unscales it. 
    It works for both min-max scaling and percentile scaling, and the minimums and maximums have
    to be passed as arguments.
    """
    unscaled_song = np.zeros((song.shape))
    for i in range(song.shape[1]):
        unscaled_song[:,i] = (song[:,i]*(max_total[i]-min_total[i])+min_total[i])
    return unscaled_song


def unstandardize(song, means, stds):
    """
    This function takes as input a song generated with a Generator Network trained with
    standardized data and unstandardizes it. It takes as well the feature means and 
    standard deviations.
    """
    unstandardized_song = np.zeros((song.shape))
    for i in range(song.shape[1]):
        unstandardized_song[:,i] = (song[:,i]*stds[i]+means[i])
    return unstandardized_song