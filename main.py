import spacy
import torch
from torchtext import data
import dill
import pickle
import torch.optim as optim
from models.rnn import *
from sklearn.metrics import classification_report
import numpy as np

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.Text).squeeze(1)

        loss = criterion(predictions, batch.Label)

        acc = binary_accuracy(predictions, batch.Label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.Text).squeeze(1)

            loss = criterion(predictions, batch.Label)

            acc = binary_accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def f_score(model, test_iterator):
    model.eval()
    predictions_list = []
    y_list = []
    with torch.no_grad():
        for batch in test_iterator:
            predictions = model(batch.Text).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            predictions_list.append(rounded_preds.view(1,-1))
            y_list.append(batch.Label.view(1,-1))
    predictions_tensor = torch.cat(predictions_list, dim=1)
    y_tensor = torch.cat(y_list, dim=1)
    predictions_tensor = predictions_tensor.view(-1)
    y_tensor = y_tensor.view(-1)

    predictions_numpy = predictions_tensor.cpu().numpy()
    y_numpy = y_tensor.cpu().numpy()
    print(classification_report(y_numpy, predictions_numpy))

def main():

    # Get data
    batch_size = 32 # It is running out of memory when we use 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = get_data(batch_size, device)

    # Define Model Parameters
    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    bidirectional` = True
    dropout = 0.5

    model = rnn(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

    pretrained_embeddings = TEXT.vocab.vectors


    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    torch.save(model.state_dict(), "models/model.%d" % epoch)

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


print('Begin training')
main()
print('End training')
