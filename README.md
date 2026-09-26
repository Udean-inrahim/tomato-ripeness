# PRD: Aplikasi Deteksi Kematangan Tomat — Flutter Project

Aplikasi Flutter untuk mendeteksi kematangan tomat (**Mentah**, **Setengah Matang**, **Matang**) dari foto, lengkap dengan bounding box, confidence score, ringkasan, dan share hasil.

**Target platform (MVP):** Android (API 26+) dan Web.
**iOS:** direncanakan menyusul (folder `ios/` sudah ada, build menyusul).

## Struktur

```
lib/
  main.dart                          # entry point + provider setup
  core/
    theme/app_theme.dart             # tema Material 3
    services/
      mock_detector.dart             # TomatoDetector interface + mock (segmentasi warna demo)
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

```bash
flutter pub get

# Android (butuh Android SDK; API 26+)
flutter run

# Web
flutter run -d chrome
# atau build statis lalu serve:
flutter build web
python -m http.server 8087 --directory build/web
```

Android APK:

```bash
flutter build apk --debug    # atau --release
```

## Mode deteksi

- **Default (mock):** segmentasi warna sederhana — bounding box mengikuti area warna tomat (merah/oranye/hijau) sebagai placeholder tanpa model. Cocok untuk uji UI & alur.
- **API (model YOLOv8 via server):** gunakan endpoint server kamu:

```bash
flutter run --dart-define=USE_API=true --dart-define=API_ENDPOINT=http://10.0.2.2:8000/detect
```

> Untuk web, ganti `10.0.2.2` dengan `localhost`.

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

## Integrasi model YOLOv8

- **On-device:** `tflite_flutter` + `yolov8n.tflite` (Android saja). Ganti `TomatoDetector` dengan implementasi TFLite, map output ke `Detection`.
- **Server-side:** Jalankan FastAPI pada `/detect` (menerima raw body `application/octet-stream`, balas JSON di atas), lalu aktifkan mode API seperti contoh di atas. Batas ukuran gambar 10 MB.