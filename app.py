import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "skinsakhi_mobilenetv3.keras")
LABEL_PATH = os.path.join(BASE_PATH, "skinsakhi_classes.txt")

THRESHOLD = 0.50
MIN_MARGIN = 0.10

model = keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Class labels do not match model outputs.")


def is_document_or_paper(image):
    rgb = np.asarray(image.convert("RGB").resize((224, 224)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    paper_ratio = np.mean(
        (gray > 215) & (hsv[:, :, 1] < 55)
    )
    edge_ratio = np.mean(cv2.Canny(gray, 60, 150) > 0)

    if paper_ratio > 0.45 and edge_ratio > 0.025:
        return True

    if paper_ratio > 0.80 and np.std(gray) < 35:
        return True

    return False


def screen_skin(image):
    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")

    if image.width < 100 or image.height < 100:
        return "Please upload a clearer image."

    if is_document_or_paper(image):
        return (
            "IMAGE NOT RELATED TO SKIN\n\n"
            "This image may contain paper, notes, or a document. "
            "Please upload a clear skin photo."
        )

    arr = np.asarray(
        image.resize((224, 224)),
        dtype=np.float32
    )[None, ...]

    predictions = model.predict(arr, verbose=0)[0]

    if not np.all(np.isfinite(predictions)):
        return "Unable to assess this image. Please try again."

    indices = np.argsort(predictions)[::-1]
    best = int(indices[0])

    confidence = float(predictions[best])
    second = float(predictions[indices[1]]) if len(indices) > 1 else 0.0

    if confidence < THRESHOLD or confidence - second < MIN_MARGIN:
        return (
            "UNABLE TO ASSESS RELIABLY\n\n"
            "Please upload a clearer image or consult a healthcare professional."
        )

    return (
        "EXPERIMENTAL MODEL SUGGESTION\n\n"
        f"Suggested class: {class_names[best]}\n\n"
        "This is not a diagnosis. The model may be wrong. "
        "Please consult a qualified healthcare professional."
    )


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue"),
    title="SkinSakhi | Skin Screening Assistant"
) as demo:
    gr.Markdown("""
    # 🩺 SkinSakhi
    ### AI-Assisted Skin Condition Screening

    Upload a clear skin photo to receive an experimental model suggestion.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Skin Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
            gr.ClearButton([image_input])

        with gr.Column():
            result_output = gr.Textbox(
                label="Screening Result",
                lines=9,
                interactive=False
            )

    analyze_btn.click(
        fn=screen_skin,
        inputs=image_input,
        outputs=result_output
    )

    gr.Markdown("""
    ---
    **Medical disclaimer:** SkinSakhi is an experimental educational
    prototype, not a medical device. It cannot diagnose or rule out
    disease. The image filter may make mistakes. Seek professional
    medical advice.
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)
