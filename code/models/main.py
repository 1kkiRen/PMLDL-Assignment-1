import os
import json
import joblib
from numpy import mean
import torch
from tqdm import tqdm
import torch.nn as nn
import huggingface_hub
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sklearn.metrics import f1_score, accuracy_score

os.environ["HF_HOME"] = "H:/.cache/huggingface"

tokens = json.load(open('api_tokens.json'))
huggingface_hub.login(token=tokens['huggingface'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ikkiren/TokenSubstitution_tokenizer')

    def __call__(self):
        return self.tokenizer


class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_size, num_classes, pad_token, tokenizer=None):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.EmbeddingBag(input_size, 128, pad_token)
        self.linear = nn.Linear(128, num_classes)
        self.tokenizer = tokenizer

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        return x


def label_preprocess(label):
    return label


def create_dataloader(dataset, tokenizer: PreTrainedTokenizerFast, batch_size=64):
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

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


def train(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs=50):
    epochs = tqdm(range(num_epochs))

    min_test_loss = 20

    try:
        for epoch in epochs:
            model.train()
            epochs.set_description(f'Epoch {epoch+1}/{num_epochs}')
            train_data = tqdm(train_dataloader)

            train_losses = []

            for batch in train_data:
                optimizer.zero_grad()
                text = batch['text'].to(device)
                label = batch['label'].to(device)
                output = model(text)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_data.set_postfix(
                    {'train_loss': format(loss.item(), '.6f')})

            model.eval()

            last_val_batch = []
            val_data = tqdm(val_dataloader)

            test_losses = []
            test_accuracies = []
            test_f1 = []

            with torch.no_grad():
                for batch in val_data:
                    text = batch['text'].to(device)
                    label = batch['label'].to(device)
                    output = model(text)
                    loss = criterion(output, label)

                    test_losses.append(loss.item())

                    last_val_batch.append(output)
                    last_val_batch.append(label)

                    ans = torch.argmax(output, dim=1)
                    test_accuracies.append(
                        accuracy_score(label.cpu(), ans.cpu()))
                    test_f1.append(
                        f1_score(label.cpu(), ans.cpu(), average='weighted'))

                    val_data.set_postfix({'test_loss': format(loss.item(), '.6f'), 'accuracy': format(
                        mean(test_accuracies), '.6f'), 'f1': format(mean(test_f1), '.6f')})

            print(f"Epoch {epoch + 1}\nTrain loss: {mean(train_losses)} Test loss: {mean(test_losses)} Test accuracy: {mean(test_accuracies)} Test F1: {mean(test_f1)}")

            if mean(test_losses) < min_test_loss:
                torch.save(model.state_dict(), 'model_linear.pth')
                print("Model saved")

            min_test_loss = min(min_test_loss, mean(test_losses))

    except KeyboardInterrupt:
        print("Training stopped")
        torch.save(model.state_dict(), 'model_linear_keyborad_stop.pth')


def test(model, test_dataloader):
    model.eval()
    test_data = tqdm(test_dataloader)

    test_accuracies = []
    test_f1 = []

    with torch.no_grad():
        for batch in test_data:
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            output = model(text)

            ans = torch.argmax(output, dim=1)
            test_accuracies.append(
                accuracy_score(label.cpu(), ans.cpu()))
            test_f1.append(
                f1_score(label.cpu(), ans.cpu(), average='weighted'))

            test_data.set_postfix(
                {'accuracy': format(mean(test_accuracies), '.6f'), 'f1': format(mean(test_f1), '.6f')})

    print(f"Test accuracy: {mean(test_accuracies)} Test F1: {mean(test_f1)}")


def main():
    dataset = load_dataset('ikkiren/multilingual-sentiments')

    tokenizer = AutoTokenizer.from_pretrained(
        'ikkiren/TokenSubstitution_tokenizer')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        dataset, tokenizer)

    model = SentimentAnalysisModel(
        len(tokenizer), 3, tokenizer.pad_token_id, tokenizer).to("cpu")

    try:
        model.load_state_dict(torch.load('model.pth'))
        print("Model loaded")
    except:
        print("Model not loaded")
        print("Training new model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train(model, optimizer, criterion, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'model.pth')

    test(model, test_dataloader)

    joblib.dump(model, open('model.pkl', 'wb'))
    joblib.dump(tokenizer, open('tokenizer.pkl', 'wb'))


if __name__ == '__main__':
    main()
