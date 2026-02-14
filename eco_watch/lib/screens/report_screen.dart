import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:geolocator/geolocator.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../services/api_service.dart';
import '../models/report_model.dart';
import '../utils/constants.dart';
import '../utils/helpers.dart';
import 'result_screen.dart';

class ReportScreen extends StatefulWidget {
  const ReportScreen({super.key});

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  File? _image;
  XFile? _webImage;
  Position? _position;
  bool _uploading = false;
  bool _gettingLoc = false;
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _getLoc();
  }

  Future<void> _getLoc() async {
    setState(() => _gettingLoc = true);
    try {
      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.whileInUse ||
          perm == LocationPermission.always) {
        _position = await Geolocator.getCurrentPosition();
      }
    } catch (_) {}
    if (mounted) setState(() => _gettingLoc = false);
  }

  Future<void> _pick(ImageSource src) async {
    final f = await _picker.pickImage(
      source: src,
      imageQuality: 80,
      maxWidth: 1024,
    );
    if (f != null) {
      setState(() {
        _webImage = f;
        _image = File(f.path);
      });
    }
  }

  Future<void> _submit() async {
    if (_image == null && _webImage == null) {
      Helpers.snack(context, 'Take a photo first', error: true);
      return;
    }

    setState(() => _uploading = true);

    if (AppProvider.useMockData) {
      await Future.delayed(const Duration(seconds: 2));
      final mockResponse = ReportResponse(
        status: "accepted",
        plasticDetected: true,
        message: "AI Analysis Complete: Plastic found.",
        detections: [
          DetectionItem(plasticType: "Plastic Bottle", confidence: 0.94),
          DetectionItem(plasticType: "Polythene Bag", confidence: 0.82),
        ],
        totalItems: 2,
        pointsEarned: 20,
      );
      setState(() => _uploading = false);
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: mockResponse)),
        );
      }
      return;
    }

    if (_position == null) {
      Helpers.snack(context, 'Location not available', error: true);
      setState(() => _uploading = false);
      return;
    }

    final userId =
        Provider.of<AppProvider>(context, listen: false).user?.userId ?? '';
    final result = await ApiService.submitReport(
      image: _image!,
      latitude: _position!.latitude,
      longitude: _position!.longitude,
      userId: userId,
    );

    setState(() => _uploading = false);

    if (result['success'] && mounted) {
      final response = ReportResponse.fromJson(result['data']);
      Provider.of<AppProvider>(context, listen: false).refreshAfterReport();
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(result: response)),
      );
    } else if (mounted) {
      Helpers.snack(context, result['error'] ?? 'Failed', error: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text(
          'Report Waste',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview
            Container(
              height: 320,
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.border, width: 2),
              ),
              child: _webImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(18),
                      child: kIsWeb
                          ? Image.network(_webImage!.path, fit: BoxFit.cover)
                          : Image.file(_image!, fit: BoxFit.cover),
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.add_a_photo_outlined,
                          size: 64,
                          color: AppColors.textSecondary.withOpacity(0.6),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Tap below to take a photo',
                          style: TextStyle(
                            color: AppColors.textSecondary,
                            fontSize: 16,
                          ),
                        ),
                      ],
                    ),
            ),
            const SizedBox(height: 24),

            // Camera / Gallery
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppColors.textPrimary,
                      side: const BorderSide(color: AppColors.border),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),

            _infoTile(
              Icons.location_on,
              _position != null
                  ? '${_position!.latitude.toStringAsFixed(4)}, ${_position!.longitude.toStringAsFixed(4)}'
                  : (_gettingLoc
                        ? 'Fetching location...'
                        : 'Location unavailable'),
              AppColors.danger,
            ),
            const SizedBox(height: 12),
            _infoTile(
              Icons.access_time_filled,
              Helpers.formatDate(DateTime.now().toIso8601String()),
              AppColors.warning,
            ),

            const SizedBox(height: 32),

            SizedBox(
              height: 60,
              child: ElevatedButton.icon(
                onPressed: _uploading ? null : _submit,
                icon: _uploading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Icon(Icons.cloud_upload, size: 28),
                label: Text(
                  _uploading ? 'Analyzing Waste...' : 'Submit Report',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoTile(IconData icon, String text, Color iconColor) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Icon(icon, color: iconColor),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: AppColors.textPrimary,
                fontWeight: FontWeight.w500,
                fontSize: 15,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
