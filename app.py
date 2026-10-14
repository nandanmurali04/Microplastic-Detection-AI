
import subprocess
import sys
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "train" / "microplastic_test" / "weights" / "best.pt"
DETECT = ROOT / "yolov5" / "detect.py"
TEST_IMAGES = ROOT / "datasets" / "images" / "test"

st.set_page_config(
    page_title="Microplastic AI | Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f 0%, #101b2d 65%, #0b1724 100%);
    color: #edf4ff;
}
[data-testid="stSidebar"] {
    background: #0b1422;
    border-right: 1px solid #23364a;
}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px;}
h1, h2, h3 {color: #f0f7ff !important;}
p, label, li {color: #c2cfdf;}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14243a, #101c2e);
    border: 1px solid #29415a;
    border-radius: 15px;
    padding: 17px;
}
div[data-testid="stMetricLabel"] {color: #9db2c9;}
div[data-testid="stMetricValue"] {color: #56e0c0;}
.hero {
    padding: 28px 30px;
    border-radius: 20px;
    border: 1px solid #29465d;
    background: linear-gradient(115deg, #12263a, #12353c);
    margin-bottom: 22px;
}
.eyebrow {
    color: #56e0c0;
    font-size: 0.78rem;
    letter-spacing: 2px;
    font-weight: 700;
}
.hero h1 {font-size: 2.3rem; margin: 8px 0;}
.hero p {color: #c0d2e2; font-size: 1.03rem;}
.panel {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #29415a;
    background: rgba(18, 35, 55, 0.8);
    margin-bottom: 12px;
}
div.stButton > button[kind="primary"] {
    background: #36cbb0;
    color: #071820;
    border: none;
    font-weight: 700;
    border-radius: 10px;
    min-height: 44px;
}
div.stButton > button[kind="primary"]:hover {
    background: #66e8cf;
    color: #071820;
}
hr {border-color: #263b50;}
.small-note {color: #91a8be; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION • RESEARCH PROTOTYPE</div>
    <h1>Microplastic AI Lab</h1>
    <p>Explore image-based particle detection using a custom-trained YOLOv5 model.</p>
</div>
""", unsafe_allow_html=True)

model_ready = WEIGHTS.exists()
if model_ready:
    model_mb = WEIGHTS.stat().st_size / (1024 * 1024)
else:
    model_mb = 0

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🔬 Microplastic AI")
    st.caption("LOCAL RESEARCH CONSOLE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Live Detector", "Dataset & Model", "About Project"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("### Model status")
    if model_ready:
        st.success("Checkpoint available")
        st.caption(f"YOLOv5s · {model_mb:.1f} MB")
    else:
        st.error("Model checkpoint missing")

    st.markdown("---")
    st.caption("Runs on your local machine")
    st.caption("Inference device: CPU")

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Project overview")
    st.write(
        "An experimental object-detection system that predicts potential "
        "microplastic locations in images and displays bounding boxes."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training images", "1,990")
    m2.metric("Validation images", "250")
    m3.metric("Test images", "250")
    m4.metric("Model", "YOLOv5s")

    st.markdown("### Initial validation results")
    a, b, c, d = st.columns(4)
    a.metric("Precision", "47.1%")
    b.metric("Recall", "41.7%")
    c.metric("mAP @ 0.5", "39.6%")
    d.metric("mAP @ 0.5:0.95", "9.48%")

    st.caption(
        "Metrics are from the initial three-epoch validation run. "
        "They are not measurements for a particular uploaded image."
    )

    st.markdown("### How the system works")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="panel"><h3>01 · Input</h3>'
                    '<p>Upload a particle image or choose a test sample.</p></div>',
                    unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="panel"><h3>02 · Inference</h3>'
                    '<p>YOLOv5 predicts object locations and confidence scores.</p></div>',
                    unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="panel"><h3>03 · Results</h3>'
                    '<p>Review annotated images and download the output.</p></div>',
                    unsafe_allow_html=True)

    st.info(
        "Research note: image detection alone cannot confirm the chemical "
        "composition of a particle."
    )

# ---------- Live detector ----------
elif page == "Live Detector":
    st.subheader("Live image detector")
    st.write("Choose an image, configure inference, and inspect the result.")

    if not model_ready:
        st.error(f"Checkpoint not found: {WEIGHTS}")
        st.stop()

    source_mode = st.radio(
        "Image source",
        ["Upload an image", "Use a dataset sample"],
        horizontal=True,
    )

    image_file = None
    sample_path = None

    if source_mode == "Upload an image":
        image_file = st.file_uploader(
            "Choose JPG, JPEG or PNG",
            type=["jpg", "jpeg", "png"],
        )
    else:
        if TEST_IMAGES.exists():
            samples = sorted(TEST_IMAGES.glob("*.jpg"))
            if samples:
                sample_names = [p.name for p in samples]
                chosen = st.selectbox(
                    "Select a test image",
                    sample_names,
                    index=min(27, len(sample_names) - 1),
                )
                sample_path = TEST_IMAGES / chosen
            else:
                st.warning("No JPG samples found in the test folder.")
        else:
            st.error(f"Test folder not found: {TEST_IMAGES}")

    with st.expander("Inference settings", expanded=True):
        conf = st.slider(
            "Minimum confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Lower values show more predictions, including uncertain ones.",
        )
        imgsz = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640],
            value=640,
        )

    if st.button("Run detection", type="primary", use_container_width=True):
        if image_file is None and sample_path is None:
            st.warning("Upload an image or select a dataset sample first.")
        else:
            uid = uuid.uuid4().hex[:8]
            input_dir = ROOT / "runs" / "demo_inputs"
            output_dir = ROOT / "runs" / "demo_outputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            input_path = input_dir / f"sample_{uid}.jpg"

            if image_file is not None:
                Image.open(image_file).convert("RGB").save(input_path)
            else:
                Image.open(sample_path).convert("RGB").save(input_path)

            run_name = f"result_{uid}"
            command = [
                sys.executable,
                str(DETECT),
                "--weights", str(WEIGHTS),
                "--source", str(input_path),
                "--imgsz", str(imgsz),
                "--conf-thres", str(conf),
                "--device", "cpu",
                "--save-txt",
                "--save-conf",
                "--project", str(output_dir),
                "--name", run_name,
                "--exist-ok",
            ]

            with st.spinner("Running YOLOv5 inference on CPU..."):
                try:
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                except subprocess.TimeoutExpired:
                    st.error("Inference exceeded four minutes. Try a smaller image size.")
                    st.stop()

            if result.returncode != 0:
                st.error("Detection failed. Details:")
                st.code((result.stderr or result.stdout)[-5000:])
            else:
                output_image = output_dir / run_name / input_path.name
                label_file = output_dir / run_name / "labels" / f"{input_path.stem}.txt"

                if output_image.exists():
                    st.session_state["detected_image"] = str(output_image)
                    st.session_state["original_image"] = str(input_path)
                    st.session_state["label_file"] = str(label_file)
                    st.session_state["detect_log"] = result.stdout
                    st.session_state["conf_threshold"] = conf
                    st.session_state["img_size"] = imgsz
                    st.success("Inference completed.")
                else:
                    st.error("The annotated image was not found.")
                    st.code(result.stdout[-3000:])

    if st.session_state.get("detected_image"):
        original = Path(st.session_state["original_image"])
        detected = Path(st.session_state["detected_image"])
        labels = Path(st.session_state["label_file"])

        if original.exists() and detected.exists():
            count = 0
            scores = []
            if labels.exists():
                rows = [
                    line.split()
                    for line in labels.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                count = len(rows)
                for row in rows:
                    if len(row) >= 6:
                        try:
                            scores.append(float(row[-1]))
                        except ValueError:
                            pass

            st.markdown("### Detection summary")
            x, y, z = st.columns(3)
            x.metric("Predicted objects", count)
            y.metric(
                "Mean confidence",
                f"{sum(scores) / len(scores):.1%}" if scores else "N/A",
            )
            z.metric("Image size", f'{st.session_state["img_size"]} px')

            left, right = st.columns(2)
            with left:
                st.markdown("**Original image**")
                st.image(str(original), use_container_width=True)
            with right:
                st.markdown("**Annotated prediction**")
                st.image(str(detected), use_container_width=True)

            st.download_button(
                "Download annotated image",
                data=detected.read_bytes(),
                file_name="microplastic_detection_result.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

            with st.expander("Technical inference log"):
                st.code(st.session_state.get("detect_log", ""))

# ---------- Dataset and model ----------
elif page == "Dataset & Model":
    st.subheader("Dataset and model details")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Dataset split")
        st.write("Training: **1,990 images**")
        st.write("Validation: **250 images**")
        st.write("Testing: **250 images**")
        st.write("Classes: **1 — microplastic**")

    with b:
        st.markdown("#### Model checkpoint")
        st.write("Architecture: YOLOv5s")
        st.write(f"Checkpoint available: **{'Yes' if model_ready else 'No'}**")
        if model_ready:
            st.write(f"Checkpoint size: **{model_mb:.2f} MB**")
        st.write("Inference: CPU")
        st.write("Initial training: 3 epochs")

    st.markdown("#### Saved training artifacts")
    run_dir = ROOT / "runs" / "train" / "microplastic_test"
    if run_dir.exists():
        artifacts = [
            p for p in run_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".csv", ".pt"}
        ]
        if artifacts:
            for item in artifacts:
                st.write(f"`{item.relative_to(ROOT)}`")
        else:
            st.caption("No image, CSV or checkpoint artifacts were found.")
    else:
        st.warning("Training output directory not found.")

    st.markdown("#### Interpretation")
    st.write(
        "The initial validation mAP@0.5 is 39.6%. This is a preliminary "
        "baseline; additional training, annotation review and small-object "
        "evaluation are needed before drawing strong conclusions."
    )

# ---------- About ----------
elif page == "About Project":
    st.subheader("About this project")
    st.write(
        "The objective is to explore whether a computer-vision model can "
        "localise potential microplastic particles in labelled images."
    )

    st.markdown("#### Technology stack")
    st.markdown(
        "- **Python** — application and inference orchestration\n"
        "- **PyTorch** — deep-learning framework\n"
        "- **YOLOv5** — object detection\n"
        "- **OpenCV / Pillow** — image processing\n"
        "- **Streamlit** — local interactive dashboard"
    )

    st.markdown("#### Current limitations")
    st.markdown(
        "- Initial training used only three epochs.\n"
        "- Validation performance is limited.\n"
        "- Very small particles are difficult to localise.\n"
        "- Predictions may include false positives or missed particles.\n"
        "- Image appearance alone cannot establish polymer identity."
    )

    st.markdown("#### Future work")
    st.markdown(
        "- Compare results across longer training runs.\n"
        "- Review annotation quality and tiny-object handling.\n"
        "- Evaluate precision and recall at different thresholds.\n"
        "- Test on independent images from different sources."
    )

st.divider()
st.markdown(
    '<p class="small-note">MICROPLASTIC AI LAB · Local research prototype · '
    'Predictions are experimental and require independent validation.</p>',
    unsafe_allow_html=True,
)
