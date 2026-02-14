class CoinTransaction {
  final String type;
  final int amount;
  final String reason;
  final String timestamp;

  CoinTransaction({
    required this.type,
    required this.amount,
    required this.reason,
    required this.timestamp,
  });

  factory CoinTransaction.fromJson(Map<String, dynamic> json) {
    return CoinTransaction(
      type: json['type'] ?? 'earned',
      amount: json['amount'] ?? 0,
      reason: json['reason'] ?? '',
      timestamp: json['timestamp'] ?? '',
    );
  }

  bool get isEarned => type == 'earned';
}

class BadgeModel {
  final String name;
  final String emoji;
  final bool isEarned;
  final String? earnedAt;
  final int? coinsRequired;

  BadgeModel({
    required this.name,
    required this.emoji,
    required this.isEarned,
    this.earnedAt,
    this.coinsRequired,
  });

  factory BadgeModel.fromJson(Map<String, dynamic> json, bool earned) {
    return BadgeModel(
      name: json['name'] ?? '',
      emoji: json['emoji'] ?? '',
      isEarned: earned,
      earnedAt: json['earnedAt'],
      coinsRequired: json['coinsRequired'],
    );
  }
}
