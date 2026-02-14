import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../utils/constants.dart';
import '../widgets/loading_widget.dart';
import '../widgets/error_widget.dart';
import '../widgets/empty_widget.dart';
import '../widgets/report_card_widget.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    Provider.of<AppProvider>(context, listen: false).fetchMyReports();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Reports'),
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
      ),
      body: Consumer<AppProvider>(
        builder: (_, p, __) {
          if (p.reportsState == ViewState.loading) {
            return const LoadingWidget(message: 'Loading reports...');
          }
          if (p.reportsState == ViewState.error) {
            return AppErrorWidget(
              message: p.error,
              onRetry: () => p.fetchMyReports(),
            );
          }
          if (p.myReports.isEmpty) {
            return const EmptyWidget(
              message: 'No reports yet!\nStart reporting waste 📸',
              icon: Icons.history,
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemCount: p.myReports.length,
            itemBuilder: (_, i) => ReportCardWidget(report: p.myReports[i]),
          );
        },
      ),
    );
  }
}
