import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class Helpers {
  static String formatDate(String iso) {
    try {
      return DateFormat('dd MMM yyyy, hh:mm a').format(DateTime.parse(iso));
    } catch (_) {
      return iso;
    }
  }

  static Color riskColor(int risk) {
    if (risk >= 71) return const Color(0xFFE53935);
    if (risk >= 41) return const Color(0xFFFDD835);
    return const Color(0xFF43A047);
  }

  static void snack(BuildContext context, String msg, {bool error = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: error ? Colors.red : const Color(0xFF43A047),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
