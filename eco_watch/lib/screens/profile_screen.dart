import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../utils/constants.dart';
import 'coin_history_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  @override
  void initState() {
    super.initState();
    Provider.of<AppProvider>(context, listen: false).fetchBadges();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Profile')),
      body: Consumer<AppProvider>(
        builder: (_, p, __) {
          final u = p.user;
          if (u == null) {
            return const Center(child: Text('Not logged in'));
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                // Header
                Container(
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: AppColors.border),
                  ),
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    children: [
                      CircleAvatar(
                        radius: 40,
                        backgroundColor: AppColors.primary.withOpacity(0.2),
                        child: Text(
                          u.tierEmoji,
                          style: const TextStyle(fontSize: 40),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        u.name,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        u.tierName,
                        style: const TextStyle(
                          fontSize: 16,
                          color: AppColors.textSecondary,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 24,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: AppColors.primary.withOpacity(0.4),
                          ),
                        ),
                        child: Text(
                          '🪙 ${u.totalCoins} Green Coins',
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: AppColors.primary,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: LinearProgressIndicator(
                          value: u.progressPercent / 100,
                          minHeight: 10,
                          backgroundColor: AppColors.border,
                          valueColor: const AlwaysStoppedAnimation(
                            AppColors.primary,
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${u.coinsToNextTier} coins to reach ${u.nextTierName}',
                        style: const TextStyle(
                          fontSize: 12,
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // Stats
                Row(
                  children: [
                    _stat('📊', '${u.totalReports}', 'Reports'),
                    const SizedBox(width: 12),
                    _stat('🔥', '${u.hotspotsFound}', 'Hotspots'),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    _stat('⚡', '${u.currentStreak}', 'Streak'),
                    const SizedBox(width: 12),
                    _stat('🏆', '#${u.cityRank}', 'City Rank'),
                  ],
                ),
                const SizedBox(height: 24),

                // Badges
                const Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    '🏅 Badges',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: AppColors.textPrimary,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 16,
                  runSpacing: 16,
                  children: p.badges.map((b) {
                    return Column(
                      children: [
                        CircleAvatar(
                          radius: 28,
                          backgroundColor: b.isEarned
                              ? AppColors.primary.withOpacity(0.2)
                              : AppColors.border,
                          child: Text(
                            b.emoji,
                            style: TextStyle(
                              fontSize: 28,
                              color: b.isEarned
                                  ? null
                                  : AppColors.textSecondary.withOpacity(0.6),
                            ),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          b.name,
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: b.isEarned
                                ? FontWeight.bold
                                : FontWeight.normal,
                            color: b.isEarned
                                ? AppColors.textPrimary
                                : AppColors.textSecondary,
                          ),
                        ),
                        if (!b.isEarned && b.coinsRequired != null)
                          Text(
                            '🔒 ${b.coinsRequired}',
                            style: const TextStyle(
                              fontSize: 9,
                              color: AppColors.textSecondary,
                            ),
                          ),
                      ],
                    );
                  }).toList(),
                ),
                const SizedBox(height: 24),

                // Coin History
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const CoinHistoryScreen(),
                      ),
                    ),
                    icon: const Icon(Icons.history),
                    label: const Text('View Coin History'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppColors.primary,
                      side: const BorderSide(color: AppColors.primary),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _stat(String emoji, String val, String label) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(emoji, style: const TextStyle(fontSize: 24)),
            const SizedBox(height: 8),
            Text(
              val,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
            ),
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
