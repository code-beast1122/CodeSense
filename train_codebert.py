import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import pickle

# --- THE AMD RX 6700 XT FIX ---
import torch_directml
# This forces PyTorch to use your AMD GPU via the Windows DirectML API
device = torch_directml.device()
print(f" Initializing CodeBERT Deep Learning on AMD GPU: {device}")
# ------------------------------

df = pd.read_csv("cpp_data_cleaned_final.csv")

# Ensure no corrupted rows snuck in
df = df.dropna(subset=['time_complexity', 'code_snippet'])
df = df[df['time_complexity'].str.startswith('O(')]

le_time = LabelEncoder()
df['time_label'] = le_time.fit_transform(df['time_complexity'])
num_classes = len(le_time.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    df['code_snippet'].values, df['time_label'].values, test_size=0.3, random_state=56
)

print("Downloading/Loading CodeBERT Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

class CodeBERTTDataset(Dataset):
    def __init__(self, code_snippets, labels, tokenizer, max_len=256):
        self.code_snippets = code_snippets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, item):
        code = str(self.code_snippets[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
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

train_dataset = CodeBERTTDataset(X_train, y_train, tokenizer)
test_dataset = CodeBERTTDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class BigOCodeBERT(nn.Module):
    def __init__(self, num_classes):
        super(BigOCodeBERT, self).__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.codebert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

print("Downloading/Loading CodeBERT Model...")
model = BigOCodeBERT(num_classes).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

epochs = 5
print(f"\n Fine-Tuning CodeBERT on {len(X_train)} C++ algorithms using RX 6700 XT...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Average Loss: {running_loss/len(train_loader):.4f}")

print("\n--- Testing CodeBERT on Unseen Data ---")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "codebert_big_o_model.pth")
with open('codebert_label_encoder.pkl', 'wb') as f:
    pickle.dump(le_time, f)
    
print("Model Saved!")