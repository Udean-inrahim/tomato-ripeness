import 'dart:io';

import 'package:http/http.dart' as http;
import 'dart:convert';

import '../../models/detection.dart';
import 'mock_detector.dart';

class ApiTomatoDetector implements TomatoDetector {
  final String endpoint;
  final http.Client _client;

  ApiTomatoDetector({required this.endpoint, http.Client? client})
      : _client = client ?? http.Client();

  @override
  Future<DetectionResult> detect(String imagePath) async {
    final request = http.MultipartRequest('POST', Uri.parse(endpoint));
    request.files
        .add(await http.MultipartFile.fromPath('image', imagePath));

    final streamed =
        await _client.send(request).timeout(const Duration(seconds: 30));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw HttpException('Server error (${response.statusCode})');
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final rawDetections = body['detections'] as List<dynamic>;

    return DetectionResult(
      imagePath: imagePath,
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
