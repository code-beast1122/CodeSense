# 🧠 CodeSense AI – Asymptotic Time Complexity Prediction with CodeBERT

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CodeBERT](https://img.shields.io/badge/HuggingFace-CodeBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/codebert-base)
[![DirectML](https://img.shields.io/badge/AMD-DirectML-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://learn.microsoft.com/en-us/windows/ai/directml/dml)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)

**CodeSense AI** is a deep learning static analysis tool that predicts the **Big-O asymptotic time complexity** of C++ algorithms directly from source code. By fine-tuning **Microsoft CodeBERT** (`microsoft/codebert-base`), CodeSense analyzes algorithmic control flows, nested loops, recursions, and data structures to classify computational complexity without needing manual execution.

---

## ✨ Features

- **🎯 Asymptotic Complexity Classification**: Classifies C++ code into algorithmic complexity tiers:
  - $O(1)$ — Constant time
  - $O(\log n)$ — Logarithmic time
  - $O(n)$ — Linear time
  - $O(n \log n)$ — Linearithmic time
  - $O(V + E)$ / $O(V)$ — Graph traversal complexities
  - $O(n^2)$, $O(2^n)$, etc. — Polynomial and exponential times
- **🚦 Traffic-Light Visualizer**:
  - 🟢 **Highly Efficient**: $O(1)$, $O(\log n)$
  - 🟡 **Moderate Efficiency**: $O(n)$, $O(n \log n)$, $O(V + E)$
  - 🔴 **Computationally Expensive**: $O(n^2)$, exponential, or higher
- **⚡ Hardware Acceleration with DirectML**: Includes native support for **AMD DirectML** (`torch_directml`), enabling high-performance GPU training and inference on Windows AMD Radeon cards (e.g. RX 6700 XT) alongside CUDA and CPU fallbacks.
- **🖥️ Interactive Streamlit Interface**: Paste raw C++ code, press analyze, and view tokenization outputs, predicted class, and efficiency rating in real time.

---

## 🏗️ Architecture

```
C++ Source Code
      │
      ▼
Microsoft CodeBERT Tokenizer (max_len = 256)
      │
      ▼
Fine-Tuned CodeBERT Base Transformer
      │
      ▼
Dropout (p = 0.3) + Classification Head
      │
      ▼
Softmax Probability Distribution -> Predicted Big-O Complexity
```

---

## 📁 Repository Structure

- `app.py`: Streamlit web interface for interactive inference.
- `train_codebert.py`: PyTorch training script fine-tuning CodeBERT on C++ snippets.
- `gen_dataset.py`: Dataset generation and preparation pipeline.
- `main.py`: Command-line interface and testing utilities.
- `cpp_data_cleaned_final.csv`: Curated dataset of C++ algorithms labeled with Big-O time complexities.
- `requirement.txt`: Python package requirements.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 - 3.11 recommended
- (Optional) AMD GPU with DirectX 12 support for DirectML, or NVIDIA GPU with CUDA

### 2. Clone the Repository
```bash
git clone https://github.com/code-beast1122/CodeSense.git
cd CodeSense
```

### 3. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirement.txt
```

> [!NOTE]
> If using AMD GPUs on Windows, ensure `torch-directml` is installed:
> ```bash
> pip install torch-directml
> ```

### 4. Training the Model
To fine-tune CodeBERT on the dataset:
```bash
python train_codebert.py
```
This generates the trained weights `codebert_big_o_model_best.pth` and label encoder `codebert_label_encoder_best.pkl`.

### 5. Launch the Web Interface
```bash
streamlit run app.py
```
Paste any C++ algorithm (e.g., QuickSort, BFS/DFS, Binary Search, or nested loops) and click **Analyze Structural Complexity**.

---

## 📊 Dataset Example

| Code Snippet | Label |
| :--- | :--- |
| `int binarySearch(int arr[], int n, int x) { ... }` | `O(log n)` |
| `void bubbleSort(int arr[], int n) { for(...) for(...) }` | `O(n^2)` |
| `void dfs(int v, vector<int> adj[]) { ... }` | `O(V + E)` |

---

## 🤝 Contributing
Contributions, dataset additions, and model architecture improvements are welcome! Feel free to submit an issue or pull request.

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
