import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';

import 'package:tomato_ripeness/models/detection.dart';

void main() {
  test('counts by category', () {
    final result = DetectionResult(
      imagePath: 'x.jpg',
      imageWidth: 100,
      imageHeight: 100,
      detections: const [
        Detection(
          label: 'matang',
          confidence: 0.9,
          box: Rect.fromLTWH(0, 0, 0.2, 0.2),
        ),
        Detection(
          label: 'matang',
          confidence: 0.8,
          box: Rect.fromLTWH(0.3, 0, 0.2, 0.2),
        ),
        Detection(
          label: 'mentah',
          confidence: 0.7,
          box: Rect.fromLTWH(0.6, 0, 0.2, 0.2),
        ),
      ],
      detectedAt: DateTime(2026, 1, 1),
    );

    expect(result.total, 3);
    expect(result.countOf('matang'), 2);
    expect(result.countOf('mentah'), 1);
    expect(result.countOf('setengah_matang'), 0);
    expect(result.summaryText(), contains('Total objek: 3'));
  });
}
