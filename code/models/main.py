from math import e
import os
import json
import torch
from tqdm import tqdm
import torch.nn as nn
import huggingface_hub
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

tokens = json.load(open('api_tokens.json'))
huggingface_hub.login(token=tokens['huggingface'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sentiment Analysis Model with 3 classes


# tokenizer = AutoTokenizer.from_pretrained(
#     'ikkiren/TokenSubstitution_tokenizer')


class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.embed = nn.Embedding(input_size, 2048)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512,
                            num_layers=2, batch_first=True)

        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x[:, -1, :]

        return x


def label_preprocess(label):
    if label == 0:
        return [1.0, 0.0, 0.0]
    elif label == 1:
        return [0.0, 1.0, 0.0]
    elif label == 2:
        return [0.0, 0.0, 1.0]


def create_dataloader(dataset, tokenizer: PreTrainedTokenizerFast, batch_size=32):
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    # Tokenize the dataset
    train_dataset = train_dataset.map(lambda x: {
        'text': tokenizer.encode(
            x['text'], truncation=True, padding='max_length', max_length=2048, add_special_tokens=False),
        'label': label_preprocess(x['label'])})
    val_dataset = val_dataset.map(lambda x: {
        'text': tokenizer.encode(
            x['text'], truncation=True, padding='max_length', max_length=2048, add_special_tokens=False),
        'label': label_preprocess(x['label'])})
    test_dataset = test_dataset.map(lambda x: {
        'text': tokenizer.encode(
            x['text'], truncation=True, padding='max_length', max_length=2048, add_special_tokens=False),
        'label': label_preprocess(x['label'])})

    train_dataset.set_format(type='torch', columns=['text', 'label'])
    val_dataset.set_format(type='torch', columns=['text', 'label'])
    test_dataset.set_format(type='torch', columns=['text', 'label'])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def train(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs=10):
    epochs = tqdm(range(num_epochs))
    
    for epoch in epochs:
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            output = model(text)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epochs.set_description(f'Epoch: {epoch+1}, Training loss: {loss.item()}')
            

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                text = batch['text'].to(device)
                label = batch['label'].to(device)
                output = model(text)
                loss = criterion(output, label)
                epochs.set_description(f'Epoch: {epoch+1}, Validation loss: {loss.item()}')


def main():
    dataset = load_dataset('ikkiren/multilingual-sentiments')

    tokenizer = AutoTokenizer.from_pretrained(
        'ikkiren/TokenSubstitution_tokenizer')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        dataset, tokenizer)

    model = SentimentAnalysisModel(len(tokenizer), 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(len(tokenizer), model, optimizer, criterion)

    train(model, optimizer, criterion, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
