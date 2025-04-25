# 🚨 Real-Time Urgent-Sound Detection Project

This project implements real-time detection of urgent environmental sounds (e.g., car horns, sirens, alarms) using two neural network models: **MobileNetV3** and **YAMNet** fine-tuned on AudioSet.

---

## 📁 Repository Structure

```
shiv_capstone/
├── mobilenetv3_weights/
│   └── mobilenetv3_large_100_ra-f55367f5.pth
├── yamnet_weights/
│   └── yamnet.h5 (optional local copy)
├── sound/
│   └── Liu/
│       ├── sound_gen_filtered.py
│       ├── demo3_sound_gen_with_prioritization.py
│       ├── hrtf_with_urgent_labels.py
│       └── schedule_sound.py
├── prototype/
│   ├── real_time_urgent_detection.py (MobileNetV3)
│   └── yamnet_real_time_urgent_detection.py (YAMNet)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Clone the Repository

```bash
git clone https://github.com/your-username/shiv_capstone.git
cd shiv_capstone
```

### Set Up the Environment

```bash
python -m venv env
# Activate virtual environment
source env/bin/activate  # macOS/Linux
.\env\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 🔗 Download Pretrained Models

- **MobileNetV3:** [Download from Hugging Face](https://huggingface.co/shiertier/models/resolve/main/mobilenetv3_large_100_ra-f55367f5.pth) and save it in:

```
mobilenetv3_weights/mobilenetv3_large_100_ra-f55367f5.pth
```

- **YAMNet:** Optionally download from [TensorFlow Hub](https://tfhub.dev/google/yamnet/1) and save it as:

```
yamnet_weights/yamnet.h5
```

---

## 🎙️ Run Real-Time Detection

- **MobileNetV3:**

```bash
python prototype/real_time_urgent_detection.py
```

- **YAMNet:**

```bash
python prototype/yamnet_real_time_urgent_detection.py
```

Use `Ctrl+C` to stop the detection loop.

---

## 🔍 How It Works

### Audio Capture & Preprocessing
- Captures real-time audio (1-second segments).
- Processes audio into spectrograms (MobileNetV3) or raw waveform (YAMNet).

### Model Predictions
- **MobileNetV3:** Outputs softmax probabilities.
- **YAMNet:** Provides frame-level and clip-level predictions.

### Urgency Detection Logic
- Alerts triggered when:
  - Prediction probability ≥ 0.25 (MobileNetV3).
  - Top prediction significantly higher than second (Δ ≥ 0.20).
  - Keyword match: `"car horn", "siren", "train", "alarm", "horn"`.
  - Special override: index `588` for verified car horn.
  - YAMNet checks top 3 predictions with threshold ≥ 0.15.

---

## 🔊 Sound Generation

- High-pitch (`sine_high`) tones indicate dangerous objects (vehicles).
- Filters repetitive alerts (COOLDOWN period).
- Sequentially queues sounds to avoid overwhelming.

---

## 🛠️ Customization

Adjust parameters directly within the detection scripts:

- Thresholds (`TH1`, `TH2`).
- Urgent keyword lists.
- Audio segment duration and sample rate.

---

## ✅ Testing

- Run detection scripts and play urgent sounds (horns, alarms).
- Verify alert logs and high-pitch sound cues.

---

## ⚠️ Troubleshooting

- Missing modules:

```bash
pip install sounddevice librosa torch torchvision timm tensorflow tensorflow_hub pygame numpy scipy
```

- Permission errors on Windows:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- Ensure model weight files are correctly placed.

---

## 🌟 Future Improvements

- More refined cooldown logic.
- Enhanced prioritization (hierarchical urgency).
- Extensive real-world testing to optimize thresholds.

---
