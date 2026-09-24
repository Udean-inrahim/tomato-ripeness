import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../../models/detection.dart';
import 'mock_detector.dart';

class ApiTomatoDetector implements TomatoDetector {
  final String endpoint;
  final http.Client _client;

  ApiTomatoDetector({required this.endpoint, http.Client? client})
      : _client = client ?? http.Client();

  @override
  Future<DetectionResult> detect(Uint8List imageBytes) async {
    final response = await _client
        .post(
          Uri.parse(endpoint),
          headers: {'Content-Type': 'application/octet-stream'},
          body: imageBytes,
        )
        .timeout(const Duration(seconds: 30));

    if (response.statusCode != 200) {
      throw Exception('Server error (${response.statusCode})');
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final rawDetections = body['detections'] as List<dynamic>;

    return DetectionResult(
      imageBytes: imageBytes,
      imageWidth: body['image_width'] as int? ?? 0,
      imageHeight: body['image_height'] as int? ?? 0,
      detections: [
        for (final item in rawDetections)
          Detection.fromJson(item as Map<String, dynamic>),
      ],
      detectedAt: DateTime.now(),
    );
  }
}
