# Micrograd - A Minimal Autograd Engine

Micrograd is a simple autograd engine and neural network implementation, inspired by the original [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. This implementation supports automatic differentiation, a multi-layer perceptron (MLP), and computation graph visualization.

## Features
- **Automatic Differentiation** using a custom `Value` class
- **Neural Network Implementation** (MLP with fully connected layers)
- **Gradient Descent Optimization** for training
- **Computation Graph Visualization** using Graphviz

---

## Table of Contents
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Example Output](#example-output)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Credits](#credits)

---

## Installation
### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd micrograd_project
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies
Ensure you have Python 3 installed, then install required packages:
```bash
pip install -r requirements.txt
```
If Graphviz is not installed on your system, install it manually:
```bash
# Linux
sudo apt install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

---

## File Structure
```
micrograd_project/
│── engine.py       # Implements Value class (autograd engine)
│── nn.py           # Implements Neuron, Layer, MLP classes
│── train.py        # Training script
│── visualize.py    # Graph visualization using Graphviz
│── requirements.txt
│── README.md
│── notebooks/
│   └── micrograd.ipynb  # Jupyter Notebook for interactive exploration
```

---

## Usage
### 1️⃣ Run the Training Script
```bash
python train.py
```
This will:
✅ Train a **Multi-Layer Perceptron (MLP)**
✅ Print **loss values** over epochs
✅ Show a **loss curve** using `matplotlib`
✅ Render a **computation graph** using `graphviz`
✅ Print **final predictions**

### 2️⃣ Visualize Computation Graph
If you want to visualize the computation graph separately, ensure **Graphviz** is installed and run:
```bash
python visualize.py
```

---

## Example Output
During training, you'll see output like this:
```
Epoch 0: Loss = 3.845
Epoch 1: Loss = 2.917
...
Epoch 99: Loss = 0.134
```
And a **plot of loss over time** will be displayed.

After training, the **computation graph** will be rendered and opened automatically.

---

## Future Improvements
- Add more activation functions (ReLU, Sigmoid, etc.)
- Implement batch training
- Extend to support PyTorch-like API

---

## License
This project is open-source under the MIT License. Feel free to modify and expand it!

---

## Credits
Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.