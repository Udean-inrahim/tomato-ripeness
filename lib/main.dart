import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'core/services/api_detector.dart';
import 'core/services/mock_detector.dart';
import 'core/theme/app_theme.dart';
import 'features/home/home_screen.dart';
import 'features/onboarding/onboarding_screen.dart';

const bool kUseApi = bool.fromEnvironment('USE_API');
const String kApiEndpoint = String.fromEnvironment(
  'API_ENDPOINT',
  defaultValue: 'http://10.0.2.2:8000/detect',
);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final onboardingDone = prefs.getBool('onboarding_done') ?? false;

  final TomatoDetector detector = kUseApi
      ? ApiTomatoDetector(endpoint: kApiEndpoint)
      : MockTomatoDetector();

  runApp(
    MultiProvider(
      providers: [
        Provider<TomatoDetector>.value(value: detector),
      ],
      child: TomatoApp(startOnboarding: !onboardingDone),
    ),
  );
}

class TomatoApp extends StatelessWidget {
  final bool startOnboarding;

  const TomatoApp({super.key, required this.startOnboarding});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deteksi Kematangan Tomat',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: startOnboarding
          ? const OnboardingScreen()
          : const HomeScreen(),
    );
  }
}
