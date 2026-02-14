class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000/api',
  );

  static String get login => '$baseUrl/auth/login';
  static String get register => '$baseUrl/auth/register';
  static String get submitReport => '$baseUrl/reports';
  static String myReports(String userId) => '$baseUrl/reports/user/$userId';
  static String get zones => '$baseUrl/zones';
  static String zoneById(String id) => '$baseUrl/zones/$id';
  static String userProfile(String id) => '$baseUrl/users/$id';
  static String userCoins(String id) => '$baseUrl/users/$id/coins';
  static String userBadges(String id) => '$baseUrl/users/$id/badges';
  static String userStreak(String id) => '$baseUrl/users/$id/streak';
  static String coinHistory(String id) => '$baseUrl/users/$id/coins/history';
  static String get leaderboard => '$baseUrl/leaderboard';
}
