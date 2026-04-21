import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # ViT Attention Explorer
    ### *An Image is Worth 16x16 Words, Dosovitskiy et al. 2020*

    Vision Transformers (ViT) changed computer vision forever by asking a simple question:
    **what if we treated an image like a sentence?**

    Instead of using convolutions, ViT splits an image into 16×16 pixel patches and processes
    them like words in a sentence using a Transformer, the same architecture behind GPT and BERT.

    This notebook lets you **see inside the model's mind** exploring exactly where it looks
    when it processes an image, across every layer and every attention head.

    ---
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from transformers import ViTModel, ViTImageProcessor
    from PIL import Image
    import requests
    from io import BytesIO

    return (
        BytesIO,
        Image,
        ViTImageProcessor,
        ViTModel,
        cm,
        mo,
        np,
        plt,
        requests,
        torch,
    )


@app.cell
def _(ViTImageProcessor, ViTModel):
    model_name = "google/vit-base-patch16-224"

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, output_attentions=True)

    model.eval()
    print("Model loaded!")
    return model, processor


@app.cell
def _(BytesIO, Image, requests):
    url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()   # good practice

    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((224, 224))

    image
    return (image,)


@app.cell
def _(image, model, processor, torch):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions

    print(f" Got {len(attentions)} layers, each shape: {attentions[0].shape}")
    return (attentions,)


@app.cell
def _(attentions, np):
    def get_attention_map(layer_idx, head_idx):
        layer = attentions[layer_idx]   # shape: [1, heads, tokens, tokens]

        # CLS token attention to image patches
        if head_idx == 12:   # mean over all heads
            attn = layer[0].mean(dim=0)[0, 1:]
        else:
            attn = layer[0, head_idx, 0, 1:]

        attn_map = attn.reshape(14, 14).cpu().numpy()

        # normalize safely
        denom = attn_map.max() - attn_map.min()
        if denom > 0:
            attn_map = (attn_map - attn_map.min()) / denom
        else:
            attn_map = np.zeros_like(attn_map)

        return attn_map

    return (get_attention_map,)


@app.cell
def _(Image, cm, get_attention_map, image, np, plt):
    def render(layer_idx, head_idx):
        attn_map = get_attention_map(layer_idx, head_idx)
        heatmap = np.uint8(cm.hot(attn_map)[:, :, :3] * 255)
        attn_img = Image.fromarray(heatmap).resize((224, 224))
        blended = Image.blend(image, attn_img, alpha=0.6)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(attn_img)
        axes[1].set_title(f"Layer {layer_idx}, Head {'avg' if head_idx==12 else head_idx}")
        axes[1].axis("off")
        axes[2].imshow(blended)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        plt.tight_layout()
        plt.close(fig)
        return fig

    return (render,)


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Attention Explorer

    Use the sliders to explore how attention changes across **layers** and **heads**.

    - **Layer** — early layers (0–3) capture low-level features like edges and textures.
    Later layers (8–11) capture high-level semantics like objects and their parts.
    - **Head** — each head learns to focus on something different.
    Head 12 averages them all.

    Drag the sliders and watch the model's focus shift in real time.
    """)
    return


@app.cell
def _(mo):
    layer_slider = mo.ui.slider(0, 11, value=11, label="Layer (0–11)")
    head_slider = mo.ui.slider(0, 12, value=12, label="Head (12 = average all)")

    mo.hstack([layer_slider, head_slider])
    return head_slider, layer_slider


@app.cell
def _(head_slider, layer_slider, render):
    render(layer_slider.value, head_slider.value)
    return


@app.cell
def _(Image, cm, get_attention_map, image, np, plt):
    def render_head_gallery(layer_idx):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
    
        for head_idx in range(12):
            attn_map = get_attention_map(layer_idx, head_idx)
            heatmap = np.uint8(cm.hot(attn_map)[:, :, :3] * 255)
            attn_img = Image.fromarray(heatmap).resize((224, 224))
            blended = Image.blend(image, attn_img, alpha=0.55)
            axes[head_idx].imshow(blended)
            axes[head_idx].set_title(f"Head {head_idx}", fontsize=11)
            axes[head_idx].axis("off")
    
        plt.suptitle(f"All 12 Attention Heads — Layer {layer_idx}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.close(fig)
        return fig

    return (render_head_gallery,)


@app.cell
def _(mo):
    mo.md("""
    ## Head Personality Gallery

    One of the most fascinating properties of Transformers is that **each attention head
    specializes**, they develop distinct "personalities" during training.

    Some heads focus on the subject's outline, others on background regions, others on
    fine-grained details. The grid below shows all 12 heads simultaneously so you can
    spot these differences yourself.

    Change the layer to see how head specialization evolves through the network.
    """)
    return


@app.cell
def _(mo):
    gallery_layer_slider = mo.ui.slider(
        0, 11,
        value=11,
        label="Layer (0–11)"
    )

    gallery_layer_slider
    return (gallery_layer_slider,)


@app.cell
def _(gallery_layer_slider, mo, render_head_gallery):
    mo.vstack([
        gallery_layer_slider,
        render_head_gallery(gallery_layer_slider.value)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **Patches as tokens** — ViT proves that the Transformer architecture generalizes
    far beyond text. Images become sequences of 196 patches (14×14 grid).

    2. **Attention = interpretability** — unlike CNNs, ViT gives us direct access to
    what the model is "looking at" through its attention weights.

    3. **Head specialization** — different heads learn different visual concepts
    automatically, with no explicit supervision.

    4. **Depth matters** — early layers build local features, later layers build
    global semantic understanding. You can see this directly in the explorer above.

    ---
    *Built with [marimo](https://marimo.io) · Model: google/vit-base-patch16-224 ·
    Paper: [Dosovitskiy et al. 2020](https://arxiv.org/abs/2010.11929)*
    """)
    return


if __name__ == "__main__":
    app.run()
