class DetectionItem {
  final String plasticType;
  final double confidence;

  DetectionItem({required this.plasticType, required this.confidence});

  factory DetectionItem.fromJson(Map<String, dynamic> json) {
    return DetectionItem(
      plasticType: json['plasticType'] ?? '',
      confidence: (json['confidence'] ?? 0).toDouble(),
    );
  }
}

class ReportResponse {
  final String status;
  final bool plasticDetected;
  final List<DetectionItem> detections;
  final int totalItems;
  final int pointsEarned;
  final String message;

  ReportResponse({
    required this.status,
    required this.plasticDetected,
    required this.detections,
    required this.totalItems,
    required this.pointsEarned,
    required this.message,
  });

  factory ReportResponse.fromJson(Map<String, dynamic> json) {
    return ReportResponse(
      status: json['status'] ?? '',
      plasticDetected: json['plastic_detected'] ?? false,
      detections:
          (json['detections'] as List?)
              ?.map((d) => DetectionItem.fromJson(d))
              .toList() ??
          [],
      totalItems: json['totalItems'] ?? 0,
      pointsEarned: json['pointsEarned'] ?? 0,
      message: json['message'] ?? '',
    );
  }
}

class ReportHistoryItem {
  final String reportId;
  final String imageUrl;
  final String status;
  final String plasticType;
  final double confidence;
  final int coinsEarned;
  final String timestamp;

  ReportHistoryItem({
    required this.reportId,
    required this.imageUrl,
    required this.status,
    required this.plasticType,
    required this.confidence,
    required this.coinsEarned,
    required this.timestamp,
  });

  factory ReportHistoryItem.fromJson(Map<String, dynamic> json) {
    return ReportHistoryItem(
      reportId: json['reportId'] ?? '',
      imageUrl: json['imageUrl'] ?? '',
      status: json['status'] ?? 'pending',
      plasticType: json['plasticType'] ?? '',
      confidence: (json['confidence'] ?? 0).toDouble(),
      coinsEarned: json['coinsEarned'] ?? 0,
      timestamp: json['timestamp'] ?? '',
    );
  }
}
