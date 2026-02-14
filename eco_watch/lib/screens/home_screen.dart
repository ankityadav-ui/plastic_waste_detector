import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../utils/constants.dart';
import '../widgets/coin_badge_widget.dart';
import 'report_screen.dart';
import 'map_screen.dart';
import 'profile_screen.dart';
import 'leaderboard_screen.dart';
import 'history_screen.dart';
import 'login_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    final p = Provider.of<AppProvider>(context, listen: false);
    if (p.user != null) {
      p.fetchProfile(p.user!.userId);
    }
  }

  void _go(Widget screen) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppStrings.appName),
        actions: [
          Consumer<AppProvider>(
            builder: (_, p, __) => p.user == null
                ? const SizedBox()
                : Padding(
                    padding: const EdgeInsets.only(right: 12),
                    child: CoinBadgeWidget(
                      coins: p.user!.totalCoins,
                      tierEmoji: p.user!.tierEmoji,
                    ),
                  ),
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              Provider.of<AppProvider>(context, listen: false).logout();
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (_) => const LoginScreen()),
              );
            },
          ),
        ],
      ),
      body: Consumer<AppProvider>(
        builder: (_, p, __) {
          final u = p.user;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Welcome Card
                Container(
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: AppColors.border),
                  ),
                  padding: const EdgeInsets.all(20),
                  child: Row(
                    children: [
                      CircleAvatar(
                        radius: 28,
                        backgroundColor: AppColors.primary.withOpacity(0.2),
                        child: Text(
                          u?.tierEmoji ?? '🌱',
                          style: const TextStyle(fontSize: 26),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Welcome back,',
                              style: TextStyle(
                                color: AppColors.textSecondary,
                                fontSize: 12,
                              ),
                            ),
                            Text(
                              u?.name ?? 'Citizen',
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: AppColors.textPrimary,
                              ),
                            ),
                            if (u?.tierName != null &&
                                (u!.tierName).trim().isNotEmpty)
                              Text(
                                u.tierName,
                                style: const TextStyle(
                                  fontSize: 13,
                                  color: AppColors.textSecondary,
                                ),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // Streak Banner
                if ((u?.currentStreak ?? 0) > 0)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppColors.primary.withOpacity(0.22),
                          AppColors.surface,
                        ],
                      ),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: AppColors.primary.withOpacity(0.4),
                      ),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.local_fire_department,
                          color: Colors.orange,
                        ),
                        const SizedBox(width: 12),
                        Text(
                          '${u!.currentStreak} Day Streak!',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: AppColors.textPrimary,
                          ),
                        ),
                        const Spacer(),
                        const Text(
                          'Keep it up 🔥',
                          style: TextStyle(
                            color: AppColors.textSecondary,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                const SizedBox(height: 24),

                // Impact Stats
                const Text(
                  'Your Impact',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),

                GridView.count(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisCount: 2,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  childAspectRatio: 1.45,
                  children: [
                    _statCard(
                      icon: Icons.bar_chart,
                      value: '${u?.totalReports ?? 0}',
                      label: 'Total Reports',
                      color: Colors.purpleAccent,
                    ),
                    _statCard(
                      icon: Icons.location_on,
                      value: '${u?.hotspotsFound ?? 0}',
                      label: 'Hotspots Found',
                      color: Colors.redAccent,
                    ),
                    _statCard(
                      icon: Icons.emoji_events,
                      value: '#${u?.cityRank ?? '-'}',
                      label: 'City Rank',
                      color: Colors.amberAccent,
                    ),
                    _statCard(
                      icon: Icons.monetization_on,
                      value: '${u?.coinsToday ?? 0}',
                      label: 'Coins Today',
                      color: Colors.greenAccent,
                    ),
                  ],
                ),

                const SizedBox(height: 24),

                // Main CTA
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () => _go(const ReportScreen()),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Report Waste'),
                  ),
                ),

                const SizedBox(height: 24),

                // Explore 2x2 Grid
                const Text(
                  'Explore',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),

                GridView.count(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisCount: 2,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  childAspectRatio: 1.4,
                  children: [
                    _menuBlock(
                      Icons.map,
                      'Nearby Hotspots',
                      'View high-risk zones',
                      () => _go(const MapScreen()),
                    ),
                    _menuBlock(
                      Icons.leaderboard,
                      'Leaderboard',
                      'Compete with others',
                      () => _go(const LeaderboardScreen()),
                    ),
                    _menuBlock(
                      Icons.history,
                      'History',
                      'Your past reports',
                      () => _go(const HistoryScreen()),
                    ),
                    _menuBlock(
                      Icons.person,
                      'Profile',
                      'Badges & Settings',
                      () => _go(const ProfileScreen()),
                    ),
                  ],
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  // Stat Card
  Widget _statCard({
    required IconData icon,
    required String value,
    required String label,
    required Color color,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 22),
          const Spacer(),
          Text(
            value,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              color: AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  // Explore Blocks (2 per row)
  Widget _menuBlock(
    IconData icon,
    String title,
    String subtitle,
    VoidCallback onTap,
  ) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.border),
        ),
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppColors.background,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: AppColors.primary, size: 20),
            ),
            const SizedBox(height: 12),
            Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 14,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontSize: 11,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
