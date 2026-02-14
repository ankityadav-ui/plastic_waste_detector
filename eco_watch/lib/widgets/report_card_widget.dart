import 'package:flutter/material.dart';
import '../models/report_model.dart';
import '../utils/helpers.dart';

class ReportCardWidget extends StatelessWidget {
  final ReportHistoryItem report;
  const ReportCardWidget({super.key, required this.report});

  Color get _statusColor {
    switch (report.status) {
      case 'verified':
        return Colors.green;
      case 'rejected':
        return Colors.red;
      default:
        return Colors.orange;
    }
  }

  IconData get _statusIcon {
    switch (report.status) {
      case 'verified':
        return Icons.check_circle;
      case 'rejected':
        return Icons.cancel;
      default:
        return Icons.pending;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: ListTile(
        leading: Icon(_statusIcon, color: _statusColor, size: 32),
        title: Text(
          report.plasticType.isNotEmpty ? report.plasticType : 'Processing...',
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Text(Helpers.formatDate(report.timestamp)),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (report.coinsEarned > 0)
              Text(
                '+${report.coinsEarned} 🪙',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF43A047),
                ),
              ),
            Text(
              report.status.toUpperCase(),
              style: TextStyle(
                fontSize: 10,
                color: _statusColor,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
