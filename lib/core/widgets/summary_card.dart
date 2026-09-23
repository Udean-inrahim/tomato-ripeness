import 'package:flutter/material.dart';

import '../../models/detection.dart';

class CategoryBadge extends StatelessWidget {
  final String label;
  final int count;
  final bool compact;

  const CategoryBadge({
    super.key,
    required this.label,
    required this.count,
    this.compact = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = Ripeness.color(label);
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 10 : 14,
        vertical: compact ? 6 : 10,
      ),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(
            '${Ripeness.display(label)} · $count',
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w600,
              fontSize: compact ? 13 : 14,
            ),
          ),
        ],
      ),
    );
  }
}

class SummaryCard extends StatelessWidget {
  final int total;
  final Map<String, int> counts;

  const SummaryCard({super.key, required this.total, required this.counts});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.insights, size: 20),
                const SizedBox(width: 8),
                Text(
                  'Ringkasan',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                ),
                const Spacer(),
                Text(
                  '$total objek',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final label in Ripeness.all)
                  CategoryBadge(label: label, count: counts[label] ?? 0),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
