import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../utils/constants.dart';
import '../utils/helpers.dart';
import '../widgets/loading_widget.dart';
import '../widgets/error_widget.dart';
import '../widgets/empty_widget.dart';

class CoinHistoryScreen extends StatefulWidget {
  const CoinHistoryScreen({super.key});

  @override
  State<CoinHistoryScreen> createState() => _CoinHistoryScreenState();
}

class _CoinHistoryScreenState extends State<CoinHistoryScreen> {
  @override
  void initState() {
    super.initState();
    Provider.of<AppProvider>(context, listen: false).fetchCoinHistory();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Coin History'),
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
      ),
      body: Consumer<AppProvider>(
        builder: (_, p, __) {
          if (p.coinHistoryState == ViewState.loading) {
            return const LoadingWidget(message: 'Loading history...');
          }
          if (p.coinHistoryState == ViewState.error) {
            return AppErrorWidget(
              message: p.error,
              onRetry: () => p.fetchCoinHistory(),
            );
          }
          if (p.coinHistory.isEmpty) {
            return const EmptyWidget(
              message: 'No transactions yet',
              icon: Icons.account_balance_wallet,
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.symmetric(vertical: 8),
            itemCount: p.coinHistory.length,
            itemBuilder: (_, i) {
              final t = p.coinHistory[i];
              return Card(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: t.isEarned
                        ? Colors.green.withOpacity(0.1)
                        : Colors.red.withOpacity(0.1),
                    child: Icon(
                      t.isEarned ? Icons.add_circle : Icons.remove_circle,
                      color: t.isEarned ? Colors.green : Colors.red,
                    ),
                  ),
                  title: Text(
                    t.reason,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  subtitle: Text(Helpers.formatDate(t.timestamp)),
                  trailing: Text(
                    '${t.isEarned ? '+' : ''}${t.amount} 🪙',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color: t.isEarned ? Colors.green : Colors.red,
                    ),
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
