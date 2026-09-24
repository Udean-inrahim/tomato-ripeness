import 'dart:typed_data';

import 'package:flutter/material.dart';

class Ripeness {
  static const mentah = 'mentah';
  static const setengahMatang = 'setengah_matang';
  static const matang = 'matang';

  static const all = [mentah, setengahMatang, matang];

  static String display(String label) {
    switch (label) {
      case mentah:
        return 'Mentah';
      case setengahMatang:
        return 'Setengah Matang';
      case matang:
        return 'Matang';
      default:
        return label;
    }
  }

  static Color color(String label) {
    switch (label) {
      case mentah:
        return const Color(0xFF43A047);
      case setengahMatang:
        return const Color(0xFFFB8C00);
      case matang:
        return const Color(0xFFE53935);
      default:
        return const Color(0xFF757575);
    }
  }
}

class Detection {
  final String label;
  final double confidence;
  final Rect box;

  const Detection({
    required this.label,
    required this.confidence,
    required this.box,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      box: Rect.fromLTWH(
        (json['x'] as num).toDouble(),
        (json['y'] as num).toDouble(),
        (json['w'] as num).toDouble(),
        (json['h'] as num).toDouble(),
      ),
    );
  }
}

class DetectionResult {
  final Uint8List imageBytes;
  final int imageWidth;
  final int imageHeight;
  final List<Detection> detections;
  final DateTime detectedAt;

  const DetectionResult({
    required this.imageBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.detections,
    required this.detectedAt,
  });

  int get total => detections.length;

  int countOf(String label) =>
      detections.where((d) => d.label == label).length;

  Map<String, int> countsByCategory() => {
        for (final label in Ripeness.all) label: countOf(label),
      };

  String summaryText() {
    final buffer = StringBuffer('Hasil Deteksi Kematangan Tomat\n');
    buffer.writeln('Total objek: $total');
    for (final label in Ripeness.all) {
      buffer.writeln('${Ripeness.display(label)}: ${countOf(label)}');
    }
    buffer.writeln('Waktu: $detectedAt');
    return buffer.toString().trimRight();
  }
}
