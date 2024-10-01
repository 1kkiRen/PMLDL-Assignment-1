import torch
import torch.nn as nn
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import uvicorn

# create an instance of FastAPI
app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


@app.get('/')
def read_root():
    return {'message': 'Welcome to the text classification app'}

# create a post method


@app.post('/predict')
def predict(payload: PredictionRequest):
    with torch.no_grad():
        tokens = tokenizer.encode(payload.text, add_special_tokens=False)
        output = model(torch.tensor(tokens).unsqueeze(0))

    ans = torch.argmax(output, dim=1)[0]

    if ans == 0:
        ans = 'Negative'
    elif ans == 1:
        ans = 'Neutral'
    else:
        ans = 'Positive'

    # return the prediction
    return {'prediction': ans}


# # run the app
if __name__ == '__main__':
    class SentimentAnalysisModel(nn.Module):
        def __init__(self, input_size, num_classes, pad_token, tokenizer):
            super(SentimentAnalysisModel, self).__init__()
            self.embedding = nn.EmbeddingBag(input_size, 128, pad_token)
            self.linear = nn.Linear(128, num_classes)
            self.tokenizer = tokenizer

        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            x = torch.softmax(x, dim=1)
            return x

    model = joblib.load(open('model.pkl', 'rb'))
    tokenizer = joblib.load(open('tokenizer.pkl', 'rb'))

    model.eval()
    model.to('cpu')
    
    uvicorn.run(app, host='0.0.0.0', port=8000)
