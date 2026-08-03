from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD


MODEL_PATH = Path("models/waste_classifier.onnx")
CLASSES_PATH = Path("artifacts/classes.json")
IMAGE_SIZE = 224


st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered",
)


@st.cache_resource
def load_model() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model was not found at: {MODEL_PATH}"
        )

    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )


@st.cache_data
def load_classes() -> list[str]:
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class file was not found at: {CLASSES_PATH}"
        )

    data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return data["classes"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    tensor = transform(image.convert("RGB"))
    tensor = tensor.unsqueeze(0)

    return tensor.numpy().astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    return exponentials / np.sum(exponentials)


def predict(
    session: ort.InferenceSession,
    image: Image.Image,
    classes: list[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    input_array = preprocess_image(image)

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: input_array},
    )

    logits = outputs[0][0]
    probabilities = softmax(logits)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    ranked_predictions = sorted(
        zip(classes, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    return predicted_class, confidence, ranked_predictions


st.title("♻️ Smart Waste Classification")
st.write(
    "Upload an image of waste and the optimized ONNX model "
    "will classify it into one of six categories."
)

st.divider()

with st.sidebar:
    st.header("Model Information")
    st.write("**Architecture:** MobileNetV3-Small")
    st.write("**Framework:** PyTorch")
    st.write("**Deployment format:** ONNX")
    st.write("**Inference engine:** ONNX Runtime")
    st.write("**Input size:** 224 × 224")
    st.write("**Classes:** 6")

    st.divider()

    st.caption(
        "This application demonstrates an end-to-end computer "
        "vision and Edge AI workflow."
    )


try:
    model_session = load_model()
    class_names = load_classes()
except Exception as error:
    st.error(str(error))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
)


if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(
        uploaded_image,
        caption="Selected waste image",
        use_container_width=True,
    )

    with st.spinner("Classifying image..."):
        prediction, confidence, ranked_predictions = predict(
            model_session,
            uploaded_image,
            class_names,
        )

    st.divider()
    st.subheader("Prediction")

    result_column, confidence_column = st.columns(2)

    with result_column:
        st.metric(
            label="Waste Category",
            value=prediction.title(),
        )

    with confidence_column:
        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%",
        )

    st.progress(float(confidence))

    st.subheader("Class Probabilities")

    for class_name, probability in ranked_predictions:
        st.write(
            f"**{class_name.title()}** — "
            f"{probability * 100:.2f}%"
        )
        st.progress(float(probability))

    st.info(
        "Predictions should be interpreted as a demonstration of the "
        "trained model and not as a replacement for industrial waste "
        "sorting systems."
    )

else:
    st.info(
        "Upload an image to begin classification. Supported formats: "
        "JPG, JPEG, PNG and WEBP."
    )


st.divider()
st.caption(
    "Built with PyTorch, MLflow, ONNX Runtime and Streamlit."
)