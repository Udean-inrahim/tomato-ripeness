import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';

import '../../core/widgets/detection_overlay.dart';
import '../../core/widgets/summary_card.dart';
import '../../models/detection.dart';

class ResultScreen extends StatelessWidget {
  final DetectionResult result;

  const ResultScreen({super.key, required this.result});

  Future<void> _share() async {
    await Share.shareXFiles(
      [XFile(result.imagePath)],
      text: result.summaryText(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final counts = result.countsByCategory();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Hasil Deteksi'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_outlined),
            tooltip: 'Bagikan hasil',
            onPressed: result.total == 0 ? null : _share,
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            if (result.total == 0)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
                      Icon(Icons.search_off,
                          size: 48,
                          color: Theme.of(context).colorScheme.outline),
                      const SizedBox(height: 12),
                      const Text('Tidak ada tomat terdeteksi di gambar ini.'),
                    ],
                  ),
                ),
              )
            else ...[
              DetectionOverlay(
                imagePath: result.imagePath,
                detections: result.detections,
                imageWidth: result.imageWidth,
                imageHeight: result.imageHeight,
              ),
              const SizedBox(height: 16),
              SummaryCard(total: result.total, counts: counts),
              const SizedBox(height: 16),
              Text(
                'Detail Objek (${result.total})',
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              for (var i = 0; i < result.detections.length; i++)
                _DetectionTile(index: i, detection: result.detections[i]),
            ],
            const SizedBox(height: 24),
            OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Pilih Ulang'),
            ),
          ],
        ),
      ),
    );
  }
}

class _DetectionTile extends StatelessWidget {
  final int index;
  final Detection detection;

  const _DetectionTile({required this.index, required this.detection});

  @override
  Widget build(BuildContext context) {
    final color = Ripeness.color(detection.label);
    final percent = (detection.confidence * 100).toStringAsFixed(1);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        child: Row(
          children: [
            CircleAvatar(
              radius: 14,
              backgroundColor: color.withValues(alpha: 0.15),
              child: Text(
                '${index + 1}',
                style: TextStyle(
                  color: color,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: color,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        Ripeness.display(detection.label),
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const Spacer(),
                      Text(
                        '$percent%',
                        style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: detection.confidence,
                      minHeight: 6,
backgroundColor: color.withValues(alpha: 0.15),
                      valueColor: AlwaysStoppedAnimation(color),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
