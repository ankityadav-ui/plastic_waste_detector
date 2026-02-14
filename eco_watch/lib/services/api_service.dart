import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../config/api_config.dart';

class ApiService {
  static Future<Map<String, dynamic>> _get(String url) async {
    try {
      final response = await http
          .get(Uri.parse(url), headers: {'Content-Type': 'application/json'})
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return {'success': true, 'data': jsonDecode(response.body)};
      }
      return {'success': false, 'error': 'Error: ${response.statusCode}'};
    } catch (e) {
      return {'success': false, 'error': 'Connection failed'};
    }
  }

  static Future<Map<String, dynamic>> _post(
    String url,
    Map<String, dynamic> body,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse(url),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200 || response.statusCode == 201) {
        return {'success': true, 'data': jsonDecode(response.body)};
      }
      return {'success': false, 'error': 'Error: ${response.statusCode}'};
    } catch (e) {
      return {'success': false, 'error': 'Connection failed'};
    }
  }

  static Future<Map<String, dynamic>> submitReport({
    required File image,
    required double latitude,
    required double longitude,
    required String userId,
  }) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse(ApiConfig.submitReport),
      );

      request.fields['latitude'] = latitude.toString();
      request.fields['longitude'] = longitude.toString();
      request.fields['userId'] = userId;
      request.fields['timestamp'] = DateTime.now().toIso8601String();
      request.files.add(await http.MultipartFile.fromPath('image', image.path));

      var streamed = await request.send().timeout(const Duration(seconds: 30));
      var response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200 || response.statusCode == 201) {
        return {'success': true, 'data': jsonDecode(response.body)};
      }
      return {'success': false, 'error': 'Upload failed'};
    } catch (e) {
      return {'success': false, 'error': 'Upload failed'};
    }
  }

  static Future<Map<String, dynamic>> login(String email, String password) =>
      _post(ApiConfig.login, {'email': email, 'password': password});

  static Future<Map<String, dynamic>> register(
    String name,
    String email,
    String password,
  ) => _post(ApiConfig.register, {
    'name': name,
    'email': email,
    'password': password,
  });

  static Future<Map<String, dynamic>> getUserProfile(String userId) =>
      _get(ApiConfig.userProfile(userId));

  static Future<Map<String, dynamic>> getUserCoins(String userId) =>
      _get(ApiConfig.userCoins(userId));

  static Future<Map<String, dynamic>> getUserBadges(String userId) =>
      _get(ApiConfig.userBadges(userId));

  static Future<Map<String, dynamic>> getUserStreak(String userId) =>
      _get(ApiConfig.userStreak(userId));

  static Future<Map<String, dynamic>> getCoinHistory(String userId) =>
      _get(ApiConfig.coinHistory(userId));

  static Future<Map<String, dynamic>> getMyReports(String userId) =>
      _get(ApiConfig.myReports(userId));

  static Future<Map<String, dynamic>> getZones() => _get(ApiConfig.zones);

  static Future<Map<String, dynamic>> getZoneDetails(String id) =>
      _get(ApiConfig.zoneById(id));

  static Future<Map<String, dynamic>> getLeaderboard() =>
      _get(ApiConfig.leaderboard);
}
