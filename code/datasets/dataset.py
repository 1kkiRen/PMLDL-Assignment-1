import os
import json
import pandas as pd
import huggingface_hub
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel


tokens = json.load(open('api_tokens.json'))
os.environ["HF_HOME"] = "D:/.cache/huggingface"
huggingface_hub.login(token=tokens['huggingface'])


def change_labels(item):
    if item['label'] == 0:
        item['label'] = 2
    elif item['label'] == 2:
        item['label'] = 0

    return item


def main():
    russian_dataset = load_dataset(
        'ai-forever/kinopoisk-sentiment-classification')
    mixed_dataset = load_dataset('tyqiangz/multilingual-sentiments', "all")

    columns_to_drop_ru = ['id', 'label_text']
    columns_to_drop_mixed = ['source', 'language']

    train_ru = russian_dataset['train']
    train_mixed = mixed_dataset['train']
    train_ru = train_ru.remove_columns(columns_to_drop_ru)
    train_mixed = train_mixed.remove_columns(columns_to_drop_mixed)
    train_mixed = train_mixed.map(change_labels)

    val_ru = russian_dataset['validation']
    val_mixed = mixed_dataset['validation']
    val_ru = val_ru.remove_columns(columns_to_drop_ru)
    val_mixed = val_mixed.remove_columns(columns_to_drop_mixed)
    val_mixed = val_mixed.map(change_labels)

    test_ru = russian_dataset['test']
    test_mixed = mixed_dataset['test']
    test_ru = test_ru.remove_columns(columns_to_drop_ru)
    test_mixed = test_mixed.remove_columns(columns_to_drop_mixed)
    test_mixed = test_mixed.map(change_labels)

    class_labels = ClassLabel(names=["negative", "neutral", "positive"])

    train_ru = train_ru.cast_column("label", class_labels)
    val_ru = val_ru.cast_column("label", class_labels)
    test_ru = test_ru.cast_column("label", class_labels)

    train_mixed = train_mixed.cast_column("label", class_labels)
    val_mixed = val_mixed.cast_column("label", class_labels)
    test_mixed = test_mixed.cast_column("label", class_labels)

    train = concatenate_datasets([train_ru, train_mixed])
    val = concatenate_datasets([val_ru, val_mixed])
    test = concatenate_datasets([test_ru, test_mixed])
    
    train = train.shuffle(seed=42)
    val = val.shuffle(seed=42)
    test = test.shuffle(seed=42)

    dataset = DatasetDict({
        'train': train,
        'validation': val,
        'test': test
    })

    dataset.push_to_hub('multilingual-sentiments', private=False)
    dataset.save_to_disk('datasets/multilingual-sentiments')


if __name__ == "__main__":
    main()
