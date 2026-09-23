import 'dart:io';
import 'dart:math';

import 'package:flutter/painting.dart';

import '../../models/detection.dart';

abstract class TomatoDetector {
  Future<DetectionResult> detect(String imagePath);
}

class MockTomatoDetector implements TomatoDetector {
  final Random _random = Random();

  @override
  Future<DetectionResult> detect(String imagePath) async {
    await Future.delayed(const Duration(milliseconds: 1500));

    final bytes = await File(imagePath).readAsBytes();
    final image = await decodeImageFromList(bytes);
    final width = image.width;
    final height = image.height;
    image.dispose();

    final count = 4 + _random.nextInt(9);
    final detections = <Detection>[];
    final used = <Rect>[];

    var attempts = 0;
    while (detections.length < count && attempts < count * 30) {
      attempts++;
      final w = 0.12 + _random.nextDouble() * 0.18;
      final h = w * (1.1 + _random.nextDouble() * 0.4);
      final left = _random.nextDouble() * (1.0 - w);
      final top = _random.nextDouble() * (1.0 - h);
      final rect = Rect.fromLTWH(left, top, w, h);

      final overlaps = used.any((r) => _overlap(r, rect) > 0.35);
      if (overlaps) continue;

      used.add(rect);
      detections.add(
        Detection(
          label: Ripeness.all[_random.nextInt(Ripeness.all.length)],
          confidence: 0.62 + _random.nextDouble() * 0.36,
          box: rect,
        ),
      );
    }

    detections.sort((a, b) {
      final ax = a.box.center.dx;
      final bx = b.box.center.dx;
      if (ax == bx) return a.box.center.dy.compareTo(b.box.center.dy);
      return ax.compareTo(bx);
    });

    return DetectionResult(
      imagePath: imagePath,
      imageWidth: width,
      imageHeight: height,
      detections: detections,
      detectedAt: DateTime.now(),
    );
  }

  double _overlap(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.isEmpty) return 0;
    return intersection.width * intersection.height /
        min(a.width * a.height, b.width * b.height);
  }
}
