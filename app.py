import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import torch_directml

# 1. HARDWARE ROUTING (Use AMD GPU for instant web predictions)
device = torch_directml.device()

# 2. REBUILD THE CODEBERT ARCHITECTURE
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

# 3. LOAD THE BRAIN (Cached so the app is lightning fast after the first load)
@st.cache_resource
def load_ai_system():
    # Load Label Encoder
    with open('codebert_label_encoder_best.pkl', 'rb') as f:
        le_time = pickle.load(f)
        
    num_classes = len(le_time.classes_)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load Trained Weights
    model = BigOCodeBERT(num_classes)
    model.load_state_dict(torch.load("codebert_big_o_model_best.pth", map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    return model, tokenizer, le_time

# --- 4. TRAFFIC LIGHT LOGIC ---
def get_complexity_color(big_o_string):
    green_classes = ['O(1)', 'O(log n)']
    yellow_classes = ['O(n)', 'O(n log n)', 'O(V + E)', 'O(V)']
    
    if any(g in big_o_string for g in green_classes):
        return "green", "🟢 Highly Efficient"
    elif any(y in big_o_string for y in yellow_classes):
        return "orange", "🟡 Moderate Efficiency"
    else:
        return "red", "🔴 Computationally Expensive"

# --- 5. THE WEB UI ---
st.set_page_config(page_title="Code-Sense AI (Transformer)", layout="centered")

st.title("🧠 AI Code-Sense Predictor")
st.markdown("Powered by **Microsoft CodeBERT**, **PyTorch**, and **AMD DirectML**.")

# Code Editor
user_code = st.text_area(
    "Paste your C++ code below:", 
    height=300, 
    placeholder="void dfs(int node) {\n    // your code here\n}"
)

if st.button("Analyze Structural Complexity", type="primary"):
    if user_code.strip() == "":
        st.warning("Please enter some C++ code to analyze.")
    else:
        with st.spinner("CodeBERT is analyzing syntax and logic structures..."):
            
            # Load systems
            model, tokenizer, le_time = load_ai_system()
            
            # Tokenize user input
            encoding = tokenizer.encode_plus(
                user_code,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, dim=1)
                
            predicted_class_id = predicted.item()
            predicted_big_o = le_time.inverse_transform([predicted_class_id])[0]
            
            # Traffic Light
            color, status_text = get_complexity_color(predicted_big_o)
            
            st.divider()
            st.subheader("Deep Learning Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Predicted Big O:")
                st.markdown(f"<h1 style='color: {color};'>{predicted_big_o}</h1>", unsafe_allow_html=True)
            with col2:
                st.markdown("### Status:")
                st.markdown(f"**{status_text}**")