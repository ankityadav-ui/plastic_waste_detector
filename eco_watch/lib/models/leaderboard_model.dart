class LeaderboardEntry {
  final int rank;
  final String userId;
  final String name;
  final int totalCoins;
  final String tier;
  final String tierEmoji;
  final int totalReports;

  LeaderboardEntry({
    required this.rank,
    required this.userId,
    required this.name,
    required this.totalCoins,
    required this.tier,
    required this.tierEmoji,
    required this.totalReports,
  });

  factory LeaderboardEntry.fromJson(Map<String, dynamic> json) {
    return LeaderboardEntry(
      rank: json['rank'] ?? 0,
      userId: json['userId'] ?? '',
      name: json['name'] ?? '',
      totalCoins: json['totalCoins'] ?? 0,
      tier: json['tier'] ?? '',
      tierEmoji: json['tierEmoji'] ?? '🌱',
      totalReports: json['totalReports'] ?? 0,
    );
  }
}
