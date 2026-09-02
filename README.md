# MicroLM — Stock Exchange Language Model

A micro GPT-style transformer language model trained on stock exchange and financial text, built from scratch with TensorFlow.

> ⚠️ **Disclaimer**: This is an educational/demo project. Generated text should **never** be used for actual trading decisions.

## Architecture

| Component          | Details                              |
|--------------------|--------------------------------------|
| Model Type         | Decoder-Only Transformer (GPT-style) |
| Embedding Dim      | 256                                  |
| Attention Heads    | 8                                    |
| Transformer Layers | 6                                    |
| FFN Hidden Dim     | 1024                                 |
| Sequence Length    | 256                                  |
| Tokenizer          | Character-level                      |
| Parameters         | ~8-12M                               |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

Training takes ~5-15 minutes depending on your hardware. With an RTX 3050, expect ~5 minutes.

### 3. Generate Text (CLI)
```bash
# Single prompt
python generate.py --prompt "The stock market" --length 500

# Interactive mode
python generate.py --interactive
```

### 4. Launch Web Demo
```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Project Structure

```
Stock-Exchange-Language-Model/
├── data/
│   ├── __init__.py
│   └── corpus.py              # Financial text corpus + HuggingFace download
├── model/
│   ├── __init__.py
│   ├── config.py              # Model hyperparameters
│   ├── tokenizer.py           # Character-level tokenizer
│   └── transformer.py         # Transformer model architecture
├── templates/
│   └── index.html             # Web UI template
├── static/
│   ├── style.css              # Dark theme styling
│   └── script.js              # Frontend logic
├── train.py                   # Training pipeline
├── generate.py                # CLI text generation
├── app.py                     # Flask web server
├── requirements.txt           # Python dependencies
└── README.md
```

## Training Data

The model trains on a combination of:
1. **Embedded corpus**: Comprehensive stock exchange text covering trading, market analysis, financial instruments, and more
2. **HuggingFace data**: Financial PhraseBank dataset (automatically downloaded)

## Customization

Adjust hyperparameters via command-line:
```bash
python train.py --epochs 100 --batch-size 32 --lr 1e-3 --embed-dim 512 --num-layers 8
```

## Tech Stack

- **TensorFlow / Keras** — Model building and training
- **Flask** — Web server
- **Chart.js** — Training metrics visualization
- **HuggingFace Datasets** — Financial data download

## License

MIT
