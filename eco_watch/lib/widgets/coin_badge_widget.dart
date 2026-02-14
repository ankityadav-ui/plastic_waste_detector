import 'package:flutter/material.dart';

class CoinBadgeWidget extends StatelessWidget {
  final int coins;
  final String tierEmoji;

  const CoinBadgeWidget({
    super.key,
    required this.coins,
    required this.tierEmoji,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(tierEmoji, style: const TextStyle(fontSize: 18)),
          const SizedBox(width: 6),
          Text(
            '🪙 $coins',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}
