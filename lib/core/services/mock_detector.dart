import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/painting.dart';
import '../../models/detection.dart';

abstract class TomatoDetector {
  Future<DetectionResult> detect(Uint8List imageBytes);
}

class MockTomatoDetector implements TomatoDetector {
  @override
  Future<DetectionResult> detect(Uint8List imageBytes) async {
    await Future.delayed(const Duration(milliseconds: 1200));

    final codec = await ui.instantiateImageCodec(
      imageBytes,
      targetWidth: 256,
    );
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final width = image.width;
    final height = image.height;
    final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    final pixels = data!.buffer.asUint8List();

    final detections = _findTomatoes(pixels, width, height);

    return DetectionResult(
      imageBytes: imageBytes,
      imageWidth: width,
      imageHeight: height,
      detections: detections,
      detectedAt: DateTime.now(),
    );
  }

  List<Detection> _findTomatoes(Uint8List pixels, int width, int height) {
    const gridX = 48;
    const gridY = 48;
    final labels = List<String?>.filled(gridX * gridY, null);

    for (var gy = 0; gy < gridY; gy++) {
      for (var gx = 0; gx < gridX; gx++) {
        final px = ((gx + 0.5) * width / gridX).floor().clamp(0, width - 1);
        final py = ((gy + 0.5) * height / gridY).floor().clamp(0, height - 1);
        final i = (py * width + px) * 4;
        labels[gy * gridX + gx] =
            _classify(pixels[i], pixels[i + 1], pixels[i + 2]);
      }
    }

    final visited = List<bool>.filled(gridX * gridY, false);
    final detections = <Detection>[];

    for (var i = 0; i < visited.length; i++) {
      if (visited[i] || labels[i] == null) continue;
      final cells = _floodFill(labels, visited, gridX, gridY, i % gridX, i ~/ gridX);

      if (cells.length < 6) continue;

      var minX = gridX, minY = gridY, maxX = 0, maxY = 0;
      var countRipe = 0;
      var countHalf = 0;
      var countRaw = 0;
      for (final cell in cells) {
        final cx = cell % gridX;
        final cy = cell ~/ gridX;
        minX = min(minX, cx);
        minY = min(minY, cy);
        maxX = max(maxX, cx);
        maxY = max(maxY, cy);
        switch (labels[cell]) {
          case Ripeness.matang:
            countRipe++;
          case Ripeness.setengahMatang:
            countHalf++;
          case Ripeness.mentah:
            countRaw++;
        }
      }

      if (minX == 0 || minY == 0 || maxX == gridX - 1 || maxY == gridY - 1) {
        continue;
      }

      final bw = maxX - minX + 1;
      final bh = maxY - minY + 1;

      final elongation = max(bw, bh) / max(bh, bw);
      if (elongation > 2.6) continue;

      final fill = cells.length / (bw * bh);
      if (fill < 0.35) continue;

      final circularity = _circularity(labels, gridX, gridY, cells);
      if (circularity < 0.50) continue;

      final areaW = bw / gridX;
      final areaH = bh / gridY;
      final covered = areaW * areaH;
      if (covered < 0.0015) continue;
      if (covered > 0.5) continue;

      final dominant = countRipe >= countHalf && countRipe >= countRaw
          ? Ripeness.matang
          : countHalf >= countRaw
              ? Ripeness.setengahMatang
              : Ripeness.mentah;

      final areaSize = covered;
      detections.add(
        Detection(
          label: dominant,
          confidence: 0.55 + min(0.4, areaSize * 8 + circularity * 0.15),
          box: Rect.fromLTRB(
            minX / gridX,
            minY / gridY,
            minX / gridX + areaW,
            minY / gridY + areaH,
          ),
        ),
      );
    }

    detections.sort((a, b) {
      final ax = a.box.center.dx;
      final bx = b.box.center.dx;
      if (ax == bx) return a.box.center.dy.compareTo(b.box.center.dy);
      return ax.compareTo(bx);
    });

    return detections.length > 18 ? detections.sublist(0, 18) : detections;
  }

  double _circularity(
    List<String?> labels,
    int gridX,
    int gridY,
    List<int> cells,
  ) {
    final inCluster = List<bool>.filled(gridX * gridY, false);
    for (final cell in cells) {
      inCluster[cell] = true;
    }

    var area = cells.length;
    var perimeter = 0;
    for (final cell in cells) {
      final cx = cell % gridX;
      final cy = cell ~/ gridX;
      for (final (nx, ny) in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]) {
        if (nx < 0 || ny < 0 || nx >= gridX || ny >= gridY) {
          perimeter++;
          continue;
        }
        if (!inCluster[ny * gridX + nx]) perimeter++;
      }
    }

    if (perimeter == 0) return 0;
    return 4 * pi * area / (perimeter * perimeter);
  }

  List<int> _floodFill(
    List<String?> labels, List<bool> visited, int gridX, int gridY,
    int startX, int startY,
  ) {
    final startLabel = labels[startY * gridX + startX]!;
    final result = <int>[];
    final stack = <(int, int)>[(startX, startY)];

    while (stack.isNotEmpty) {
      final (x, y) = stack.removeLast();
      if (x < 0 || y < 0 || x >= gridX || y >= gridY) continue;
      final idx = y * gridX + x;
      if (visited[idx] || labels[idx] != startLabel) continue;
      visited[idx] = true;
      result.add(idx);
      stack.add((x + 1, y));
      stack.add((x - 1, y));
      stack.add((x, y + 1));
      stack.add((x, y - 1));
    }

    return result;
  }

  String? _classify(int r, int g, int b) {
    final (h, s, v) = _rgbToHsv(r, g, b);

    if (v < 0.15 || v > 0.95 || s < 0.20) return null;

    if (h <= 22 || h >= 340) {
      return Ripeness.matang;
    }
    if (h >= 18 && h <= 60 && s >= 0.30) {
      return Ripeness.setengahMatang;
    }
    if (h >= 65 && h <= 160 && s >= 0.25) {
      return Ripeness.mentah;
    }
    return null;
  }

  (double, double, double) _rgbToHsv(int rawR, int rawG, int rawB) {
    final r = rawR / 255.0;
    final g = rawG / 255.0;
    final b = rawB / 255.0;

    final maxC = max(r, max(g, b));
    final minC = min(r, min(g, b));
    final delta = maxC - minC;

    double h;
    if (delta == 0) {
      h = 0;
    } else if (maxC == r) {
      h = 60 * (((g - b) / delta) % 6);
    } else if (maxC == g) {
      h = 60 * ((b - r) / delta + 2);
    } else {
      h = 60 * ((r - g) / delta + 4);
    }
    if (h < 0) h += 360;

    final s = maxC == 0 ? 0.0 : delta / maxC;
    return (h, s, maxC);
  }
}
