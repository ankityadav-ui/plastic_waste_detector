import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_model.dart';
import '../models/zone_model.dart';
import '../models/leaderboard_model.dart';
import '../models/report_model.dart';
import '../models/coin_model.dart';
import '../services/api_service.dart';

enum ViewState { idle, loading, success, error }

class AppProvider extends ChangeNotifier {
  // 🔥 TOGGLE THIS: true for Demo/Judges, false for Real Backend
  static const bool useMockData = true;

  UserModel? _user;
  UserModel? get user => _user;
  String? _token;
  String? get token => _token;

  List<ZoneModel> _zones = [];
  List<ZoneModel> get zones => _zones;
  ViewState _zonesState = ViewState.idle;
  ViewState get zonesState => _zonesState;

  List<LeaderboardEntry> _leaderboard = [];
  List<LeaderboardEntry> get leaderboard => _leaderboard;
  ViewState _leaderboardState = ViewState.idle;
  ViewState get leaderboardState => _leaderboardState;

  List<ReportHistoryItem> _myReports = [];
  List<ReportHistoryItem> get myReports => _myReports;
  ViewState _reportsState = ViewState.idle;
  ViewState get reportsState => _reportsState;

  List<BadgeModel> _badges = [];
  List<BadgeModel> get badges => _badges;

  List<CoinTransaction> _coinHistory = [];
  List<CoinTransaction> get coinHistory => _coinHistory;
  ViewState _coinHistoryState = ViewState.idle;
  ViewState get coinHistoryState => _coinHistoryState;

  String _error = '';
  String get error => _error;

  // ==================== AUTH ====================

  Future<bool> login(String email, String password) async {
    if (useMockData) {
      _user = _mockUser;
      _token = "mock_token_123";
      notifyListeners();
      return true;
    }
    final result = await ApiService.login(email, password);
    if (result['success']) {
      _user = UserModel.fromJson(result['data']['user'] ?? result['data']);
      _token = result['data']['token'];
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('userId', _user!.userId);
      notifyListeners();
      return true;
    }
    _error = result['error'] ?? 'Login failed';
    notifyListeners();
    return false;
  }

  // 🔥 ADDED THIS METHOD TO FIX YOUR ERROR
  Future<bool> register(String name, String email, String password) async {
    if (useMockData) {
      _user = _mockUser;
      _token = "mock_token_123";
      notifyListeners();
      return true;
    }
    final result = await ApiService.register(name, email, password);
    if (result['success']) {
      _user = UserModel.fromJson(result['data']['user'] ?? result['data']);
      _token = result['data']['token'];
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('userId', _user!.userId);
      notifyListeners();
      return true;
    }
    _error = result['error'] ?? 'Registration failed';
    notifyListeners();
    return false;
  }

  Future<void> loadSavedUser() async {
    if (useMockData) {
      _user = _mockUser;
      notifyListeners();
      return;
    }
    final prefs = await SharedPreferences.getInstance();
    final userId = prefs.getString('userId');
    if (userId != null) await fetchProfile(userId);
  }

  void logout() {
    _user = null;
    notifyListeners();
  }

  // ==================== FETCH METHODS ====================

  Future<void> fetchZones() async {
    _zonesState = ViewState.loading;
    notifyListeners();
    if (useMockData) {
      await Future.delayed(const Duration(milliseconds: 500));
      _zones = _mockZones;
      _zonesState = ViewState.success;
    } else {
      final result = await ApiService.getZones();
      if (result['success']) {
        _zones = (result['data']['zones'] as List)
            .map((z) => ZoneModel.fromJson(z))
            .toList();
        _zonesState = ViewState.success;
      } else {
        _zonesState = ViewState.error;
      }
    }
    notifyListeners();
  }

  Future<void> fetchLeaderboard() async {
    _leaderboardState = ViewState.loading;
    notifyListeners();
    if (useMockData) {
      _leaderboard = _mockLeaderboard;
      _leaderboardState = ViewState.success;
    } else {
      final result = await ApiService.getLeaderboard();
      if (result['success']) {
        _leaderboard = (result['data']['leaderboard'] as List)
            .map((l) => LeaderboardEntry.fromJson(l))
            .toList();
        _leaderboardState = ViewState.success;
      } else {
        _leaderboardState = ViewState.error;
      }
    }
    notifyListeners();
  }

  Future<void> fetchMyReports() async {
    _reportsState = ViewState.loading;
    notifyListeners();
    if (useMockData) {
      _myReports = _mockReports;
      _reportsState = ViewState.success;
    } else {
      if (_user == null) return;
      final result = await ApiService.getMyReports(_user!.userId);
      if (result['success']) {
        _myReports = (result['data']['reports'] as List)
            .map((r) => ReportHistoryItem.fromJson(r))
            .toList();
        _reportsState = ViewState.success;
      } else {
        _reportsState = ViewState.error;
      }
    }
    notifyListeners();
  }

  Future<void> fetchBadges() async {
    if (useMockData) {
      _badges = _mockBadges;
    } else {
      if (_user == null) return;
      final result = await ApiService.getUserBadges(_user!.userId);
      if (result['success']) {
        _badges = [];
        for (var b in (result['data']['earned'] as List? ?? []))
          _badges.add(BadgeModel.fromJson(b, true));
        for (var b in (result['data']['locked'] as List? ?? []))
          _badges.add(BadgeModel.fromJson(b, false));
      }
    }
    notifyListeners();
  }

  Future<void> fetchProfile(String userId) async {
    if (!useMockData) await ApiService.getUserCoins(userId);
  }

  Future<void> fetchCoinHistory() async {
    if (useMockData) _coinHistory = [];
    notifyListeners();
  }

  Future<void> refreshAfterReport() async {
    if (!useMockData) await fetchProfile(_user!.userId);
  }

  // ==================== MOCK DATA ====================

  final UserModel _mockUser = UserModel(
    userId: "user_ankit",
    name: "Ankit Kumar",
    email: "ankit@hackathon.com",
    totalCoins: 750,
    tierName: "Green Champion",
    tierEmoji: "🌳",
    cityRank: 3,
    totalReports: 47,
    hotspotsFound: 8,
    currentStreak: 12,
    longestStreak: 20,
    progressPercent: 75,
    coinsToNextTier: 250,
    nextTierName: "Sustainability Star",
    coinsToday: 40,
    coinsThisWeek: 180,
  );

  final List<ZoneModel> _mockZones = [
    ZoneModel(
      zoneId: "Z-01",
      name: "Sector 62 Park",
      latitude: 0,
      longitude: 0,
      riskScore: 85,
      totalReports: 42,
    ),
    ZoneModel(
      zoneId: "Z-02",
      name: "Central Market",
      latitude: 0,
      longitude: 0,
      riskScore: 92,
      totalReports: 120,
    ),
    ZoneModel(
      zoneId: "Z-03",
      name: "Tech University",
      latitude: 0,
      longitude: 0,
      riskScore: 30,
      totalReports: 5,
    ),
  ];

  final List<LeaderboardEntry> _mockLeaderboard = [
    LeaderboardEntry(
      rank: 1,
      userId: "u1",
      name: "Rahul Sharma",
      totalCoins: 3200,
      tier: "Eco Legend",
      tierEmoji: "👑",
      totalReports: 150,
    ),
    LeaderboardEntry(
      rank: 2,
      userId: "u2",
      name: "Priya Verma",
      totalCoins: 1800,
      tier: "Sustainability Star",
      tierEmoji: "⭐",
      totalReports: 80,
    ),
    LeaderboardEntry(
      rank: 3,
      userId: "user_ankit",
      name: "Ankit (You)",
      totalCoins: 750,
      tier: "Green Champion",
      tierEmoji: "🌳",
      totalReports: 47,
    ),
  ];

  final List<ReportHistoryItem> _mockReports = [
    ReportHistoryItem(
      reportId: "r1",
      imageUrl: "",
      status: "verified",
      plasticType: "Plastic Bottle",
      confidence: 0.94,
      coinsEarned: 20,
      timestamp: "2025-01-15T10:30:00Z",
    ),
    ReportHistoryItem(
      reportId: "r2",
      imageUrl: "",
      status: "verified",
      plasticType: "Polythene Bag",
      confidence: 0.88,
      coinsEarned: 20,
      timestamp: "2025-01-14T14:20:00Z",
    ),
  ];

  final List<BadgeModel> _mockBadges = [
    BadgeModel(name: "Beginner", emoji: "🌱", isEarned: true),
    BadgeModel(name: "Eco Warrior", emoji: "🌿", isEarned: true),
    BadgeModel(name: "Green Champion", emoji: "🌳", isEarned: true),
  ];
}
