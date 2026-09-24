import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../../models/detection.dart';

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Rect? sourceRect;

  DetectionPainter(this.detections, {this.sourceRect});

  @override
  void paint(Canvas canvas, Size size) {
    final src = sourceRect ?? const Rect.fromLTWH(0, 0, 1, 1);

    for (final detection in detections) {
      final rect = Rect.fromLTWH(
        (detection.box.left * src.width + src.left) * size.width,
        (detection.box.top * src.height + src.top) * size.height,
        detection.box.width * src.width * size.width,
        detection.box.height * src.height * size.height,
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
      oldDelegate.detections != detections ||
      oldDelegate.sourceRect != sourceRect;
}

class DetectionOverlay extends StatefulWidget {
  final Uint8List imageBytes;
  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  const DetectionOverlay({
    super.key,
    required this.imageBytes,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  State<DetectionOverlay> createState() => _DetectionOverlayState();
}

class _DetectionOverlayState extends State<DetectionOverlay> {
  Size? _decodedSize;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  Future<void> _decode() async {
    try {
      final codec = await ui.instantiateImageCodec(widget.imageBytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      if (mounted) {
        setState(() {
          _decodedSize = Size(image.width.toDouble(), image.height.toDouble());
        });
      }
      image.dispose();
    } catch (_) {
      if (mounted) setState(() => _failed = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final aspect = _decodedSize?.width == null
        ? (widget.imageWidth > 0 && widget.imageHeight > 0
            ? widget.imageWidth / widget.imageHeight
            : 4 / 3)
        : _decodedSize!.width / _decodedSize!.height;

    return AspectRatio(
      aspectRatio: aspect,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final boxW = constraints.maxWidth;
            final boxH = constraints.maxHeight;

            Rect? display;
            final imgW = _decodedSize?.width ?? boxW;
            final imgH = _decodedSize?.height ?? boxH;
            if (imgW > 0 && imgH > 0) {
              final scale = (boxW / imgW < boxH / imgH)
                  ? boxW / imgW
                  : boxH / imgH;
              final dw = imgW * scale;
              final dh = imgH * scale;
              display = Rect.fromLTWH(
                (boxW - dw) / 2,
                (boxH - dh) / 2,
                dw,
                dh,
              );
            }

            final srcRect = display == null
                ? null
                : Rect.fromLTWH(
                    display.left / boxW,
                    display.top / boxH,
                    display.width / boxW,
                    display.height / boxH,
                  );

            return Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(
                  widget.imageBytes,
                  fit: BoxFit.contain,
                  errorBuilder: (_, __, ___) => _failed
                      ? Container(
                          color: Colors.black12,
                          child: const Icon(Icons.broken_image, size: 48),
                        )
                      : Container(color: Colors.black12),
                ),
                if (_decodedSize != null || _failed)
                  CustomPaint(
                    painter: DetectionPainter(
                      widget.detections,
                      sourceRect: srcRect,
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}
