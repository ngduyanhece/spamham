import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


class SVMClassifier:
    def __init__(self, kernel='linear'):
        self.model = SVC(kernel=kernel, random_state=42)
    
    def prepare_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, report


class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2, max_len=128, batch_size=32, lr=2e-5, epochs=1):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.label_encoder = LabelEncoder() 
    
    def load_data(self, text_data, labels):
        self.text_data = text_data
        self.labels = self.label_encoder.fit_transform(labels)
        self.texts = self.text_data.tolist()
        self.labels = self.labels.tolist()

    def create_dataloaders(self):
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        self.train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.max_len)
        self.test_dataset = TextDataset(X_test, y_test, self.tokenizer, self.max_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)

        def train_epoch(model, data_loader, optimizer, device):
            model = model.train()
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            return total_loss / len(data_loader)

        for epoch in range(self.epochs):
            train_loss = train_epoch(self.model, self.train_loader, optimizer, self.device)
            print(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_loss}')
    
    def evaluate(self):
        def eval_model(model, data_loader, device):
            model = model.eval()
            predictions, true_labels = [], []
            with torch.no_grad():
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).flatten()
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
            return predictions, true_labels

        y_pred, y_true = eval_model(self.model, self.test_loader, self.device)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return acc, report
