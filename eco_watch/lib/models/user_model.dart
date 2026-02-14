class UserModel {
  final String userId;
  final String name;
  final String email;
  final int totalCoins;
  final String tierName;
  final String tierEmoji;
  final int cityRank;
  final int totalReports;
  final int hotspotsFound;
  final int currentStreak;
  final int longestStreak;
  final double progressPercent;
  final int coinsToNextTier;
  final String nextTierName;
  final int coinsToday;
  final int coinsThisWeek;

  UserModel({
    required this.userId,
    required this.name,
    required this.email,
    required this.totalCoins,
    required this.tierName,
    required this.tierEmoji,
    required this.cityRank,
    required this.totalReports,
    required this.hotspotsFound,
    required this.currentStreak,
    required this.longestStreak,
    required this.progressPercent,
    required this.coinsToNextTier,
    required this.nextTierName,
    required this.coinsToday,
    required this.coinsThisWeek,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      userId: json['userId'] ?? '',
      name: json['name'] ?? '',
      email: json['email'] ?? '',
      totalCoins: json['totalCoins'] ?? 0,
      tierName: json['tier']?['name'] ?? 'Beginner',
      tierEmoji: json['tier']?['emoji'] ?? '🌱',
      cityRank: json['cityRank'] ?? 0,
      totalReports: json['stats']?['totalReports'] ?? 0,
      hotspotsFound: json['stats']?['hotspotsFound'] ?? 0,
      currentStreak: json['stats']?['currentStreak'] ?? 0,
      longestStreak: json['stats']?['longestStreak'] ?? 0,
      progressPercent: (json['progressPercent'] ?? 0).toDouble(),
      coinsToNextTier: json['nextTier']?['coinsNeeded'] ?? 0,
      nextTierName: json['nextTier']?['name'] ?? '',
      coinsToday: json['stats']?['coinsToday'] ?? 0,
      coinsThisWeek: json['stats']?['coinsThisWeek'] ?? 0,
    );
  }
}
