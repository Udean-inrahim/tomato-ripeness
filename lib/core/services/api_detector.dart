import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../../models/detection.dart';
import 'mock_detector.dart';

class DetectionApiException implements Exception {
  final String message;

  const DetectionApiException(this.message);

  @override
  String toString() => message;
}

class ApiTomatoDetector implements TomatoDetector {
  final String endpoint;
  final http.Client _client;

  ApiTomatoDetector({required this.endpoint, http.Client? client})
      : _client = client ?? http.Client();

  void close() => _client.close();

  @override
  Future<DetectionResult> detect(Uint8List imageBytes) async {
    final http.Response response;
    try {
      response = await _client
          .post(
            Uri.parse(endpoint),
            headers: {'Content-Type': 'application/octet-stream'},
            body: imageBytes,
          )
          .timeout(const Duration(seconds: 45));
    } on TimeoutException {
      throw const DetectionApiException(
        'Server deteksi terlalu lama. Coba lagi beberapa saat.',
      );
    } catch (_) {
      throw const DetectionApiException(
        'Tidak dapat menghubungi server. Pastikan API sedang aktif.',
      );
    }

    if (response.statusCode != 200) {
      throw DetectionApiException(_errorMessage(response));
    }

    try {
      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) {
        throw const FormatException('Respons bukan objek JSON');
      }

      final rawDetections = decoded['detections'];
      if (rawDetections is! List) {
        throw const FormatException('Field detections tidak valid');
      }

      final detections = <Detection>[];
      for (final item in rawDetections) {
        if (item is! Map<String, dynamic>) {
          throw const FormatException('Item deteksi tidak valid');
        }
        detections.add(Detection.fromJson(item));
      }

      return DetectionResult(
        imageBytes: imageBytes,
        imageWidth: (decoded['image_width'] as num?)?.toInt() ?? 0,
        imageHeight: (decoded['image_height'] as num?)?.toInt() ?? 0,
        detections: detections,
        detectedAt: DateTime.now(),
      );
    } on FormatException {
      throw const DetectionApiException('Respons server tidak valid.');
    } catch (_) {
      throw const DetectionApiException('Format detections tidak valid.');
    }
  }

  String _errorMessage(http.Response response) {
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic> && decoded['detail'] is String) {
        return decoded['detail'] as String;
      }
    } catch (_) {}
    return 'Server error (${response.statusCode}).';
  }
}
