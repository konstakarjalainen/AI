import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def get_device():
    is_cuda = torch.cuda.is_available()
    print("Using cuda? =", is_cuda)
    return torch.device("cuda" if is_cuda else "cpu")


def letter_encoder(filename):

    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    print(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_vocab = len(chars)

    zeros_arr = np.zeros(n_vocab)
    data_matrix = []
    for char in raw_text:
        zeros_copy = zeros_arr.copy()
        zeros_copy[char_to_int[char]] = 1.0
        data_matrix.append(zeros_copy)
    # summarize the loaded data

    return data_matrix, n_vocab, char_to_int, int_to_char


class CharMLP(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_vocab, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        # produce output
        x = x[:, -1, :]
        x = self.linear(x)
        return x


filename = "abcde_edcba.txt"
data_matrix, n_vocab, char_to_int, int_to_char = letter_encoder(filename)
device = get_device()
model = CharMLP().to(device)
seq_length = 2
dataX = []
dataY = []
for i in range(0, len(data_matrix)-seq_length, 1):
    seq_in = data_matrix[i:i + seq_length]
    seq_out = data_matrix[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)
print("Training data size: ", n_patterns)

# To tensors
X = torch.tensor(np.array(dataX), dtype=torch.float32, device=device).reshape(n_patterns, seq_length, n_vocab)
y = torch.tensor(np.array(dataY), dtype=torch.float32, device=device)

n_epochs = 3
batch_size = 16
learning_rate = 0.01

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)


for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # Validation
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss}")


start = np.random.randint(0, n_patterns-seq_length)
input = data_matrix[start:start+seq_length]
print()
print(int_to_char[int(input[1].argmax())], end="")

with torch.no_grad():
    for i in range(1, 50):
        x = torch.tensor(input, dtype=torch.float32, device=device).reshape(1, seq_length, n_vocab)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        res = np.zeros(n_vocab)
        res[index] = 1.0
        input.append(res)
        input = input[1:]
