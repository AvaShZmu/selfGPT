# GPT From Scratch

A hands-on exploration of how GPT (Generative Pre-trained Transformer) actually works, implemented from the ground up using PyTorch. This project aims to build an intuition on how these LLMs work from the inside. Heavy instructions from Andrej Karpathy and his video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s

## What's Inside

This repository contains two main implementations:

### 1. Bigram Model (`bigram_model.py`)
A simple baseline model that predicts the next character based only on the current character. This serves as a starting point to understand:
- Basic neural language modeling
- PyTorch training loops
- Text tokenization and data preparation

### 2. GPT Transformer (`gpt/`)
A full transformer implementation with:
- **Multi-head self-attention** (`Head` and `MultiHead` classes)
- **Feed-forward networks** with residual connections
- **Layer normalization** and **positional embeddings**
- **Training script** (`train.py`) with checkpointing support
- **Inference script** (`inference.py`) for interactive text generation

## Installation

### 1. Clone the repository
```
git clone <your-repo-url>
cd gpt-from-scratch
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Prepare your dataset
Place your text file as `gpt/input.txt`. For example, download the TinyShakespeare dataset:
```
cd gpt/
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Or use any text file you want:
```
cp /path/to/your/textfile.txt gpt/input.txt
```

## Training

### Train the GPT Model
```
cd gpt/
python train.py
```

**What happens during training:**
- Loads and tokenizes your text data
- Splits into train/validation sets (90%/10%)
- Trains for 10,000 iterations with progress tracking
- Shows train/validation loss every 500 steps
- Automatically saves the best model to `../trained_models/`
- Uses GPU if available, falls back to CPU

**Training Configuration (modify in `train.py`):**
- `n_embd`: 384 (embedding dimensions)
- `n_head`: 6 (attention heads)
- `n_layer`: 6 (transformer layers)
- `block_size`: 512 (context length)
- `batch_size`: 64
- `iterations`: 10,000
- `learning_rate`: 1e-4 with cosine scheduling

**Expected training time:**
- GPU: ~30-60 minutes
- CPU: 3-6 hours

### Resume Training from Checkpoint
To continue training from a saved model:
```python
# In train.py, set:
CHECKPOINT = "your-model-name.pth"
```

### Train the Bigram Baseline
For comparison, train the simple bigram model:
```
python bigram_model.py
```

## Text Generation (Inference)

### Interactive Text Generation
```
cd gpt/
python inference.py
```

**Interactive prompts:**
- Enter your text prompt
- Set temperature (0.1-2.0): Controls randomness
  - 0.1: Very focused, repetitive
  - 0.8: Balanced creativity
  - 1.5+: Very creative, chaotic
- Set max tokens: How much text to generate

### Example Usage
```
Enter prompt: Once upon a time
Temperature (0.1-2.0, default 0.8): 0.8
Max tokens (default 200): 300

Generated text:
Once upon a time, in a kingdom far away, there lived a brave knight...
```

## Testing and Evaluation

### Monitor Training Progress
Watch the loss values during training:
- **Train loss**: Should decrease steadily
- **Validation loss**: Should decrease but may plateau
- **Good signs**: Val loss < 2.0 for character-level models
- **Overfitting**: Train loss much lower than val loss

### Test Text Quality
Generate samples with different settings:

```
# Conservative generation
Temperature: 0.3-0.5

# Balanced generation  
Temperature: 0.7-0.9

# Creative generation
Temperature: 1.0-1.5
```

### Compare Models
1. Train bigram model: `python bigram_model.py`
2. Train GPT model: `cd gpt && python train.py`
3. Compare generated text quality

## Model Architecture

The GPT implementation (`BasicGPT`) includes:
- **Token embeddings**: Convert characters to vectors
- **Positional embeddings**: Add position information
- **6 Transformer blocks**: Each with self-attention + feed-forward
- **Layer normalization**: Stabilizes training
- **Residual connections**: Helps gradient flow
- **384-dimensional embeddings**: Rich representation space

## File Structure
```
gpt-from-scratch/
├── README.md
├── requirements.txt
├── bigram_model.py          # Simple baseline model
├── gpt/
│   ├── input.txt           # Your training data
│   ├── gpt_model.py        # GPT architecture
│   ├── train.py            # Training script
│   └── inference.py        # Text generation
└── trained_models/         # Saved models appear here
    └── your-model.pth
```

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```python
# Reduce batch_size in train.py
batch_size = 32  # or 16
```

**Poor text quality:**
- Train longer (increase `iterations`)
- Use larger dataset
- Increase model size (`n_embd`, `n_layer`)

**Training too slow:**
- Reduce model size for testing
- Use smaller dataset
- Reduce `block_size`

**Model not improving:**
- Check learning rate (try 3e-4 to 1e-3)
- Ensure dataset is large enough (>1MB text)
- Verify data quality

## Advanced Usage

### Custom Dataset
Replace `gpt/input.txt` with your own text file. Works best with:
- Books, articles, code, poetry
- At least 1MB of text
- Consistent style/domain

### Hyperparameter Tuning
Key parameters to experiment with:
- `learning_rate`: 1e-4 to 6e-4
- `n_embd`: 192, 384, 768
- `n_layer`: 4, 6, 12
- `dropout`: 0.1 to 0.3

### Model Comparison
Track different experiments:
```
# Models are saved with validation loss in filename
# Example: shakespeare-basicgpt-val-1.52.pth
# Lower validation loss = better model
```

---

*This project is perfect for understanding transformer architectures, attention mechanisms, and the fundamentals of modern language models. Start with the bigram model to understand basics, then move to the full GPT implementation.*
