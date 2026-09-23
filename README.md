# PRD: Aplikasi Deteksi Kematangan Tomat — Flutter Project

Aplikasi Android (Flutter) untuk mendeteksi kematangan tomat (**Mentah**, **Setengah Matang**, **Matang**) dari foto, lengkap dengan bounding box, confidence score, ringkasan, dan share hasil.

## Struktur

```
lib/
  main.dart                          # entry point + provider setup
  core/
    theme/app_theme.dart             # tema Material 3
    services/
      mock_detector.dart             # TomatoDetector interface + mock (on-device demo)
      api_detector.dart              # implementasi REST (FastAPI/Flask)
    widgets/
      summary_card.dart              # ringkasan per kategori
      detection_overlay.dart         # CustomPainter bounding box + label
  models/
    detection.dart                   # Detection, DetectionResult, Ripeness
  features/
    onboarding/onboarding_screen.dart
    home/home_screen.dart
    detection/detection_screen.dart  # pick kamera/galeri + analisis
    result/result_screen.dart        # overlay, ringkasan, detail, share
```

## Menjalankan

Project ini berisi source code `lib/` + `pubspec.yaml`. Platform folder di-generate sekali:

```bash
flutter create . --org com.example --project-name tomato_ripeness
flutter pub get
flutter run
```

## Mode deteksi

- **Default (mock):** deteksi simulasi on-device — UI & alur bisa diuji tanpa model.
- **API:** jalankan dengan endpoint server kamu:

```bash
flutter run --dart-define=USE_API=true --dart-define=API_ENDPOINT=http://10.0.2.2:8000/detect
```

Format respons API yang diharapkan:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    { "label": "matang", "confidence": 0.94, "x": 0.1, "y": 0.2, "w": 0.2, "h": 0.25 }
  ]
}
```

`x,y,w,h` dinormalisasi 0–1 terhadap ukuran gambar.

## Integrasi model YOLOv8 (TFLite) — opsional

Ganti `MockTomatoDetector` dengan service `tflite_flutter` yang menjalankan model YOLOv8-nano hasil export `yolov8n.tflite` (quantized), lalu map output ke `Detection`.
