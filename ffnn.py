import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        
        # [to fill] obtain output layer representation
        output = self.W2(hidden)
        
        # ensure shape for calculating softmax 
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
            
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

if __name__ == "__main__":
    hidden_dims = [50, 100, 200]
    epoch_list = [3, 5, 10]

    log_file_path = "results/experiment_log.csv"
    os.makedirs("results", exist_ok=True)

    with open(log_file_path, "w") as log_file:
        log_file.write("hidden_dim,epochs,val_accuracy\n")

        for hidden_dim in hidden_dims:
            for num_epochs in epoch_list:
                print(f"\n=== Running model with hidden_dim={hidden_dim}, epochs={num_epochs} ===")

                random.seed(42)
                torch.manual_seed(42)

                train_data, valid_data = load_data("train.json", "val.json")
                vocab = make_vocab(train_data)
                vocab, word2index, index2word = make_indices(vocab)
                train_data = convert_to_vector_representation(train_data, word2index)
                valid_data = convert_to_vector_representation(valid_data, word2index)

                model = FFNN(input_dim=len(vocab), h=hidden_dim)
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

                train_acc_history = []
                val_acc_history = []

                for epoch in range(num_epochs):
                    model.train()
                    optimizer.zero_grad()
                    random.shuffle(train_data)
                    minibatch_size = 16
                    N = len(train_data)
                    correct = 0
                    total = 0
                    for minibatch_index in range(N // minibatch_size):
                        optimizer.zero_grad()
                        loss = None
                        for example_index in range(minibatch_size):
                            input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                            predicted_vector = model(input_vector)
                            example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                            loss = example_loss if loss is None else loss + example_loss
                        (loss / minibatch_size).backward()
                        optimizer.step()

                        predicted_label = torch.argmax(predicted_vector)
                        correct += int(predicted_label == gold_label)
                        total += 1
                    train_acc = correct / total
                    train_acc_history.append(train_acc)
                    print(f"Epoch {epoch + 1}/{num_epochs} - Training Accuracy: {train_acc:.4f}")

                    # Validation after each epoch
                    model.eval()
                    correct = 0
                    total = 0
                    for input_vector, gold_label in valid_data:
                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector)
                        correct += int(predicted_label == gold_label)
                        total += 1
                    val_accuracy = correct / total
                    val_acc_history.append(val_accuracy)

                print(f"Final Validation Accuracy: {val_accuracy:.4f}")
                log_file.write(f"{hidden_dim},{num_epochs},{val_accuracy:.4f}\n")

                # Plot accuracy over epochs
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, num_epochs + 1), train_acc_history, label="Training Accuracy")
                plt.plot(range(1, num_epochs + 1), val_acc_history, label="Validation Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title(f"Hidden Dim={hidden_dim}, Epochs={num_epochs}")
                plt.legend()
                plt.grid(True)

                plot_filename = f"results/acc_plot_h{hidden_dim}_e{num_epochs}.png"
                plt.savefig(plot_filename)
                plt.close()
