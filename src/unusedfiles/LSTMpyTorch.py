import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import DataLoader

from utils.data_parser import normalize_vecframe_by_col
from utils.storage import load_frame,  check_if_vecframe, dump_frame

class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        self.decoder = nn.LSTM(self.latent_dim, self.input_dim, self.num_layers)


    def forward(self, input):

        out, (latent_state, cell_state) = self.encoder(input)
        emb = latent_state

        # This part is just for decoding
        middle_layer = latent_state.repeat(len(input), 1, 1)
        # Decode
        y, _ = self.decoder(middle_layer)
        return torch.squeeze(y), emb


epochs, batchsize = int(sys.argv[2]), int(sys.argv[3])
in_dsp_fname = str(sys.argv[1])  #'1minute_emb_dsp'


in_dsp = load_frame(in_dsp_fname) #dzne_dsp = dzne_dsp.iloc[:100, :], data_path="../../data/dzne/"


if 'nor' not in in_dsp_fname:

    normalize_vecframe_by_col(in_dsp_fname)

    in_dsp_fname = in_dsp_fname + "_nor"
    print ("nor NOT found in filename, Vecframe will be normalized and saved to ", in_dsp_fname)
    in_dsp = load_frame(in_dsp_fname)

else:

    print("nor found in filename, Not normalizing!")
     # dzne_dsp = dzne_dsp.iloc[:100, :], data_path="../../data/dzne/"

print (in_dsp.head(3))

Xtrain = in_dsp.iloc[:, 2:].values
seq_len = in_dsp.shape[1] - 2

Xtrain = torch.from_numpy(Xtrain).type(torch.Tensor)
Xtrain_tens_dl = Xtrain.view([len(Xtrain), seq_len, 1])
data_loader = DataLoader(Xtrain_tens_dl, batch_size=batchsize, shuffle=True)


model = LSTM(input_dim = 1, latent_dim = int(seq_len/4), num_layers=1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


for ep in range(epochs):

    for in_batch_dl in data_loader:

        in_batch = in_batch_dl.view([seq_len, len(in_batch_dl), 1])
        sqz_in_bat = torch.squeeze(in_batch)

        sqz_pred, emb = model(in_batch)

        optimizer.zero_grad()

        loss = loss_function(sqz_pred, sqz_in_bat)
        print("Epoch ", ep, "MSE: ", loss.item())
        loss.backward()
        optimizer.step()

    #torch.save(model.state_dict(), f = in_dsp_fname + "LSTM_" + str(epochs) + "_" + str(batchsize))

#X_in_tens = Xtrain.view([seq_len, len(Xtrain), 1 ]) #input shape(num of timesteps, batch, num_features)

embArr, outArr= [], []



for _, row in in_dsp.iterrows():

    head = [int(row[0]), int(row[1])]

    row_tens = torch.from_numpy(row.iloc[2:].values).type(torch.Tensor).view([seq_len, 1, 1])
    out, emb = model(row_tens)

    emb = torch.squeeze(emb)

    embArr.append(head + (emb.data.tolist()))
    outArr.append(head + (out.data.tolist()))


emb_df = pd.DataFrame(data = embArr, columns=['user', 'desc'] + [str(i) for i in range(len(embArr[0]) - 2)])
out_df = pd.DataFrame(data = outArr, columns=['user', 'desc'] + [str(i) for i in range(len(outArr[0]) - 2)])

print (out_df.shape, emb_df.shape)

print (out_df.head(3))
print (emb_df.head(3))

check_if_vecframe(out_df)
check_if_vecframe(emb_df)

outname = in_dsp_fname + "_LSTM_out_" + str(epochs) + "_" + str(batchsize)
embname = in_dsp_fname + "_LSTM_emb_" + str(epochs) + "_" + str(batchsize)

dump_frame(frame=out_df, name= outname, in_csv=True)
dump_frame(frame=emb_df, name= embname, in_csv=True)

print ("written emb and out to disk in ", outname, embname )

"""
NOTES


1 LSTM learns 1 sequence, 


"""
