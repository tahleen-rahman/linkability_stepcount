import os.path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset


from compress import Compressor
from utils.storage import DATA_PATH

class PytorchAE(Compressor):
    """
    Intermediate class for any Pytorch Auto Encoders
    """
    def __init__(self, vec_name, data_path=DATA_PATH, out_path=DATA_PATH, num_epochs = 10, batch_size = 100, learning_rate = 1e-3, save_model=True):
        """
        Constructor of child classes have to set:
            self.ae_class
            self.AE_name
            self.par_name
            self.ae_params

        :param vec_name: vecframe name, e.g., 'fake'
        :param data_path: path to dzne where vecframe is located
        :param num_epochs: number of epochs for network training
        :param batch_size: size of each batch for network training
        :param learning_rate: learning rate for network training
        """
        super().__init__(vec_name, data_path, out_path)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_model = save_model


    def train(self):
        model_file_name = '{}{}_{}-{}.model'.format(self.out_path, self.vec_name, self.AE_name, self.par_name)
        # if trained previously:
        if os.path.isfile(model_file_name):
            print("loading model from '{}'".format(model_file_name))
            self.model = self.ae_class(self.ae_params).float()
            self.model.load_state_dict(torch.load(model_file_name))
            self.model.eval()
        else:
            # create the model
            self.model = self.ae_class(self.ae_params).cpu().float()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

            for epoch in range(self.num_epochs):
                for data in self.data_loader:
                    data = data[0].float()
                    # ===================reshape=====================
                    data = self.reshape(data)
                    # ===================forward=====================
                    output = self.model(data)
                    loss = criterion(output, data)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.4f}'
                      .format(epoch + 1, self.num_epochs, loss.item()))
            if self.save_model:
                torch.save(self.model.state_dict(), model_file_name)

    def reshape(self, data):
        return data

    def calculate_embeddings(self):
        return [list(key) + self.model.encoder(value.float()).tolist() for key, value in self.vecs.items()]

    def compress_save(self):
        super().compress_save()
        print("prepare dataset")
        self.pytorch_prepare_dataset()
        print("training")
        self.train()
        print("making vecframe")
        embeddings = self.calculate_embeddings()
        vecframe = pd.DataFrame(embeddings, columns = ['user', 'desc'] + [str(i) for i in range(len(embeddings[0]) - 2)])
        print("saving in feather")
        self.dump_vecframe(vecframe, '{}-{}_emb'.format(self.AE_name, self.par_name), in_csv=False)

    def calculate_reconstructions(self):
        return [list(key) + self.model.forward(value.float()).tolist() for key, value in self.vecs.items()]

    def calculate_save_reconstructions(self):
        # in case we load the model just to calculate reconstructions
        if not self.data_loader:
            self.pytorch_prepare_dataset()
        if not self.model:
            self.train()
        appendix = '{}-{}_rec'.format(self.AE_name, self.par_name)
        results = self.calculate_reconstructions()
        vecframe = pd.DataFrame(results, columns = ['user', 'desc'] + [str(i) for i in range(len(results[0]) - 2)])
        self.dump_vecframe(vecframe, appendix, in_csv=False)

    def pytorch_prepare_dataset(self):
        """
        splits dataset into days and users

        requires:
         self.batch_size
        prepares:
         self.vecs - a dictionary from user, desc to pytorch tensor vectors
         self.data_loader - pytorch dataloader
        """
        self.vecs = {}
        #temp_holder = []
        for index, row in self.vecframe.iterrows():
            user = int(row[0])
            desc = int(row[1])
            vector = np.array([i for i in row[2:].to_numpy()])
            # if (vector < 0).any() or (vector > 1).any():
            #     raise Exception("Vecframe needs to be normalized first.")
            #tensor = torch.from_numpy(vector)
            #self.vecs[(user, desc)] = tensor.type(torch.DoubleTensor)
            self.vecs[(user, desc)] = torch.from_numpy(vector)

        # for day in self.stepframe.day.unique():
        #     vf = self.stepframe[self.stepframe.day == day].iloc[:, 4:]
        #     for user in range(1000):
        #         vector = np.array([i / 41 for i in vf[str(user)].values.astype(np.float32)])
        #         tensor = torch.from_numpy(vector)
        #         self.vecs[(user, day)] = tensor.type(torch.FloatTensor)
        #         #temp_holder.append(torch.from_numpy(vector))
        # prepare the data_loader
        all_vecs = torch.stack(list(self.vecs.values()))
        dataset = TensorDataset(all_vecs)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


