import 'dart:io';

import 'package:flutter/material.dart';

import '../../models/detection.dart';

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      final rect = Rect.fromLTWH(
        detection.box.left * size.width,
        detection.box.top * size.height,
        detection.box.width * size.width,
        detection.box.height * size.height,
      );

      final color = Ripeness.color(detection.label);
      final outline = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3;

      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(6)),
        outline,
      );

      final label =
          '${Ripeness.display(detection.label)} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final bgWidth = textPainter.width + 12;
      final bgHeight = textPainter.height + 8;
      final inside = rect.top - bgHeight - 4 >= 0;
      final bgTop = inside ? rect.top - bgHeight - 4 : rect.top + 4;
      final bgLeft =
          rect.left.clamp(0.0, size.width - bgWidth).toDouble();

      final bgRect = Rect.fromLTWH(bgLeft, bgTop, bgWidth, bgHeight);
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(6)),
        Paint()..color = color,
      );
      textPainter.paint(
        canvas,
        Offset(bgRect.left + 6, bgRect.top + 4),
      );
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) =>
      oldDelegate.detections != detections;
}

class DetectionOverlay extends StatelessWidget {
  final String imagePath;
  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  const DetectionOverlay({
    super.key,
    required this.imagePath,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  Widget build(BuildContext context) {
    final aspect =
        imageWidth > 0 && imageHeight > 0 ? imageWidth / imageHeight : 4 / 3;
    return AspectRatio(
      aspectRatio: aspect,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.file(
              File(imagePath),
              fit: BoxFit.fill,
              errorBuilder: (_, __, ___) => Container(
                color: Colors.black12,
                child: const Icon(Icons.broken_image, size: 48),
              ),
            ),
            CustomPaint(painter: DetectionPainter(detections)),
          ],
        ),
      ),
    );
  }
}
