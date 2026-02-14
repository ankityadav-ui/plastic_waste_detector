import 'dart:ui';

class ZoneModel {
  final String zoneId;
  final String name;
  final double latitude;
  final double longitude;
  final int riskScore;
  final int totalReports;

  ZoneModel({
    required this.zoneId,
    required this.name,
    required this.latitude,
    required this.longitude,
    required this.riskScore,
    required this.totalReports,
  });

  factory ZoneModel.fromJson(Map<String, dynamic> json) {
    return ZoneModel(
      zoneId: json['zoneId'] ?? '',
      name: json['name'] ?? '',
      latitude: (json['latitude'] ?? 0).toDouble(),
      longitude: (json['longitude'] ?? 0).toDouble(),
      riskScore: json['riskScore'] ?? 0,
      totalReports: json['totalReports'] ?? 0,
    );
  }

  Color get zoneColor {
    if (riskScore >= 71) return const Color(0xFFE53935);
    if (riskScore >= 41) return const Color(0xFFFDD835);
    return const Color(0xFF43A047);
  }

  String get statusEmoji {
    if (riskScore >= 71) return '🔴';
    if (riskScore >= 41) return '🟡';
    return '🟢';
  }

  String get statusText {
    if (riskScore >= 71) return 'Hotspot';
    if (riskScore >= 41) return 'Warning';
    return 'Clean';
  }
}
