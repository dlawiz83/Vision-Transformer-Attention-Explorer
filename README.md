# 🔬 Vision Transformer Attention Explorer

An interactive marimo notebook that visualizes how Vision Transformers (ViT) process images through multi-head attention mechanisms. Built to help researchers, students, and curious minds understand the internals of transformer-based vision models.

## 🎯 Overview

Vision Transformers revolutionized computer vision by treating images as sequences of patches and applying the Transformer architecture (originally designed for NLP) to vision tasks. This project makes the model's "attention" — *where it looks* when processing an image — visible and interactive.

**Key Innovation:** The Head Personality Gallery reveals how different attention heads specialize in different visual concepts (edges, objects, backgrounds) without explicit supervision.

## ✨ Features

- **🎛️ Interactive Attention Explorer**
  - Explore attention maps across all 12 transformer layers
  - Inspect individual attention heads or view averaged attention
  - Real-time visualization updates as you adjust sliders
  - Overlay attention heatmaps on the original image for direct interpretation

- **🧠 Head Personality Gallery**
  - View all 12 attention heads simultaneously in a 3×4 grid
  - See how head specialization evolves across network depth
  - Early layers capture low-level features (edges, textures)
  - Later layers capture high-level semantics (objects, parts)

- **📚 Educational Narrative**
  - Clear explanations of ViT architecture and attention mechanisms
  - Key takeaways connecting theory to visualization
  - Intuitive introduction for researchers and students

## 📖 Paper Implemented

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

- **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **Year:** 2020
- **ArXiv:** https://arxiv.org/abs/2010.11929

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone or download this repository
git clone https://github.com/dlawiz/Vision-Transformer-Attention-Explorer.git


# Install dependencies
pip install -r requirements.txt

# Launch the marimo notebook
marimo edit vit_explorer.py
```

The notebook will open in your browser at `http://localhost:3000`.

### Option 2: Run on molab (Cloud)

No installation needed — open the notebook directly:
👉 [Live on molab](https://molab.marimo.io/notebooks/nb_7XUNXqSwtE2hjW2bLahyRk)

## 🛠️ Tech Stack

- **marimo** — Next-generation reactive Python notebook (replacing Jupyter)
- **PyTorch** — Deep learning framework
- **Hugging Face Transformers** — Pre-trained ViT model
- **matplotlib** — Visualization
- **Pillow** — Image processing
- **NumPy** — Numerical computing

## 📊 How It Works

1. **Load pretrained ViT** — Uses `google/vit-base-patch16-224` from Hugging Face
2. **Process image** — Splits image into 14×14 = 196 patches (16×16 pixels each)
3. **Extract attention** — Captures all 12 layers × 12 heads of attention weights
4. **Visualize** — Renders attention as heatmaps overlaid on the original image
5. **Explore** — Interact with sliders to examine different layers and heads

### Understanding Attention Maps

- **Bright (yellow/white):** Regions the model pays high attention to
- **Dark (red/black):** Regions the model ignores
- **CLS Token:** The special "[CLS]" token's attention is visualized — it aggregates information about the entire image

## 🧪 What You'll Learn

- How Transformers work in vision (beyond just NLP)
- What "attention" means mathematically and visually
- How neural networks develop specialized sub-units (attention heads)
- How to interpret and debug deep learning models through attention visualization
- How to build interactive data exploration tools with marimo

## 💡 Original Extensions

Beyond reimplementing the paper, this notebook includes:

1. **Head Specialization Analysis** — Grid view of all 12 heads simultaneously to spot emergent behaviors
2. **Layer-wise progression** — Watch how attention becomes sharper and more semantic as you go deeper
3. **Interactive exploration** — Real-time updates make hypothesis testing immediate and intuitive

## 📁 Project Structure

```
Vision-Transformer-Attention-Explorer/
├── vit_explorer.py          # Main marimo notebook
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── .gitignore              # Git ignore rules
```

## ⚙️ System Requirements

- **Python:** 3.8+
- **RAM:** 8GB minimum (for model loading)
- **GPU:** Not required (CPU works fine for inference on single images)
- **OS:** macOS, Linux, or Windows

## 🔧 Installation Details

### Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

First run will download the ViT model (~350MB) automatically from Hugging Face.

## 🎮 Usage

Once the notebook is open in your browser:

1. **Explore Layer-by-Layer Attention**
   - Drag the "Layer" slider to see how attention changes across depth
   - Drag the "Head" slider to isolate individual heads (or view averaged attention)
   - Watch the three panels: Original image, pure attention map, and overlay

2. **View the Head Personality Gallery**
   - Scroll to the gallery section
   - Use the layer slider to see all 12 heads at once
   - Notice how different heads focus on different image regions

3. **Read the explanations**
   - Each section includes text explaining what you're looking at
   - Try to predict what different heads will focus on, then verify visually

## 📝 Example Insights

- **Layer 0–2:** Noisy, scattered attention across patches (learning basic features)
- **Layer 6–8:** Beginning to focus on object regions (mid-level semantics)
- **Layer 11:** Sharp, concentrated attention on the subject (high-level understanding)
- **Head 5 (often):** May suppress the main subject to focus on background context
- **Head 1–3 (often):** Focus tightly on edges and object boundaries

## 🤝 Contributing

Found a bug? Have a cool idea for an extension?

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-idea`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-idea`)
5. Open a Pull Request

## 📚 Further Reading

- **ViT Paper:** https://arxiv.org/abs/2010.11929
- **marimo docs:** https://docs.marimo.io
- **Attention is All You Need:** https://arxiv.org/abs/1706.03762 (original Transformer paper)
- **BERT paper:** https://arxiv.org/abs/1810.04805 (multi-head attention explained)

## ⚖️ License

This project is open source and available under the **MIT License**. See LICENSE file for details.

## 🙏 Acknowledgments

- **Dosovitskiy et al.** for the Vision Transformer paper
- **Hugging Face** for pre-trained models and transformers library
- **marimo team** for the incredible reactive notebook environment
- **alphaXiv** for curating accessible research papers

## 📧 Contact

Questions or feedback? Open an issue on GitHub or reach out directly.

---

**Built for the marimo × alphaXiv Notebook Competition (2026)**
