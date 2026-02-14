import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../utils/constants.dart';
import '../widgets/loading_widget.dart';
import '../widgets/error_widget.dart';
import '../widgets/empty_widget.dart';

class LeaderboardScreen extends StatefulWidget {
  const LeaderboardScreen({super.key});

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  @override
  void initState() {
    super.initState();
    Provider.of<AppProvider>(context, listen: false).fetchLeaderboard();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Leaderboard'),
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
      ),
      body: Consumer<AppProvider>(
        builder: (_, p, __) {
          if (p.leaderboardState == ViewState.loading) {
            return const LoadingWidget(message: 'Loading leaderboard...');
          }
          if (p.leaderboardState == ViewState.error) {
            return AppErrorWidget(
              message: p.error,
              onRetry: () => p.fetchLeaderboard(),
            );
          }
          if (p.leaderboard.isEmpty) {
            return const EmptyWidget(
              message: 'No contributors yet',
              icon: Icons.leaderboard,
            );
          }

          // Get current user ID to highlight them
          final myId = p.user?.userId;

          return ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: p.leaderboard.length,
            itemBuilder: (_, i) {
              final e = p.leaderboard[i];
              final isMe = e.userId == myId;
              final isTop3 = e.rank <= 3;

              return Card(
                color: isMe
                    ? AppColors.primary.withOpacity(0.1)
                    : isTop3
                    ? Colors.amber.withOpacity(0.05)
                    : null,
                margin: const EdgeInsets.only(bottom: 8),
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: isTop3 ? Colors.amber : Colors.grey[300],
                    child: Text(
                      '#${e.rank}',
                      style: TextStyle(
                        color: isTop3 ? Colors.white : Colors.black,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  title: Row(
                    children: [
                      Text(e.tierEmoji, style: const TextStyle(fontSize: 20)),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          isMe ? '${e.name} (You)' : e.name,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: isMe ? AppColors.primaryDark : null,
                          ),
                        ),
                      ),
                    ],
                  ),
                  subtitle: Text('${e.tier} • ${e.totalReports} reports'),
                  trailing: Text(
                    '🪙 ${e.totalCoins}',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color: AppColors.primaryDark,
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
