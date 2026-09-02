"""
Flask web application for MicroLM — Interactive stock exchange text generation.
"""
import json
import os

from flask import Flask, render_template, request, jsonify

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import MicroLM
from generate import load_model, generate_text

app = Flask(__name__)

# ── Global model state ──
_model = None
_tokenizer = None
_config = None
_history = None


def get_model():
    """Lazy-load the trained model."""
    global _model, _tokenizer, _config, _history

    if _model is None:
        print("[App] Loading model...")
        _model, _tokenizer, _config = load_model("checkpoints")
        print("[App] Model loaded successfully!")

        # Load training history if available
        history_path = "training_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                _history = json.load(f)
            print("[App] Training history loaded.")

    return _model, _tokenizer, _config, _history


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate text from a prompt."""
    try:
        model, tokenizer, config, _ = get_model()

        data = request.get_json()
        prompt = data.get("prompt", "The stock market")
        temperature = float(data.get("temperature", 0.8))
        top_k = int(data.get("top_k", 40))
        max_length = int(data.get("max_length", 500))

        # Clamp values
        temperature = max(0.1, min(2.0, temperature))
        top_k = max(1, min(100, top_k))
        max_length = max(50, min(2000, max_length))

        clean_prompt = prompt

        generated = generate_text(
            model, tokenizer, clean_prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        return jsonify({
            "success": True,
            "prompt": clean_prompt,
            "original_prompt": prompt,
            "generated_text": generated,
            "settings": {
                "temperature": temperature,
                "top_k": top_k,
                "max_length": max_length,
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-info")
def api_model_info():
    """Return model architecture information."""
    try:
        model, tokenizer, config, history = get_model()

        info = {
            "architecture": "GPT-style Decoder-Only Transformer",
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "ff_dim": config.ff_dim,
            "seq_length": config.seq_length,
            "parameters": f"{model.get_num_params():,}",
            "tokenizer": "Subword (BPE)",
        }

        if history:
            info["training"] = {
                "epochs": history.get("epochs_completed", 0),
                "final_loss": round(history["loss"][-1], 4) if history.get("loss") else None,
                "final_val_loss": round(history["val_loss"][-1], 4) if history.get("val_loss") else None,
                "training_time": round(history.get("training_time_seconds", 0), 1),
            }

        return jsonify({"success": True, "info": info})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/training-history")
def api_training_history():
    """Return training history for visualization."""
    try:
        _, _, _, history = get_model()

        if history:
            return jsonify({"success": True, "history": history})
        else:
            return jsonify({"success": False, "error": "No training history found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MicroLM - Stock Exchange Language Model")
    print("  Web Demo")
    print("=" * 60)
    print("\n  Starting server at http://localhost:5000\n")

    # Pre-load model
    get_model()

    app.run(host="0.0.0.0", port=5000, debug=False)
