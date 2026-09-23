import 'package:flutter/material.dart';

import '../../models/detection.dart';
import '../detection/detection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Deteksi Kematangan Tomat')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [scheme.primary, scheme.primary.withValues(alpha: 0.75)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Sortasi tomat lebih cepat & konsisten',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Ambil foto, dan aplikasi akan mendeteksi setiap tomat '
                    'beserta tingkat kematangannya.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Colors.white.withValues(alpha: 0.9),
                        ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: scheme.primary,
                      minimumSize: const Size.fromHeight(48),
                    ),
                    onPressed: () => Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const DetectionScreen(),
                      ),
                    ),
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Deteksi Tomat'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Kategori Kematangan',
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            for (final label in Ripeness.all)
              _CategoryTile(label: label),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Icon(Icons.tips_and_updates_outlined,
                        color: scheme.primary),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Tips: foto dengan pencahayaan cukup dan latar '
                        'belakang sederhana menghasilkan deteksi terbaik.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CategoryTile extends StatelessWidget {
  final String label;

  const _CategoryTile({required this.label});

  @override
  Widget build(BuildContext context) {
    final color = Ripeness.color(label);
    final description = switch (label) {
      Ripeness.mentah => 'Warna hijau, tekstur keras, belum siap panen.',
      Ripeness.setengahMatang =>
        'Warna kekuningan/jingga, mulai matang.',
      Ripeness.matang => 'Warna merah merata, siap dikonsumsi.',
      _ => '',
    };

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        title: Text(
          Ripeness.display(label),
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Text(description),
      ),
    );
  }
}
