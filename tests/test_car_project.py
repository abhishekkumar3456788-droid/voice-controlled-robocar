import unittest

from car_project import clean_words, has_word, is_hallucination, parse_args, to_command


class CommandParsingTests(unittest.TestCase):
    def test_clean_words_normalizes_punctuation_and_case(self):
        self.assertEqual(clean_words("  Turn LEFT, please! "), ["turn", "left", "please"])

    def test_recognizes_driving_commands(self):
        cases = {
            "go forward": "F",
            "reverse": "B",
            "turn left": "L",
            "turn right": "R",
            "please stop now": "S",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(to_command(phrase), expected)

    def test_stop_wins_over_other_recognized_command(self):
        self.assertEqual(to_command("go stop"), "S")

    def test_ignores_ambiguous_or_long_commands(self):
        for phrase in ("left right", "hello there", "go forward right now"):
            with self.subTest(phrase=phrase):
                self.assertIsNone(to_command(phrase))

    def test_filters_common_recognition_hallucinations(self):
        self.assertTrue(is_hallucination("Thank you for watching!"))
        self.assertTrue(is_hallucination("..."))
        self.assertFalse(is_hallucination("go forward"))

    def test_mode_words_match_exactly_or_with_small_recognition_error(self):
        self.assertTrue(has_word("please chat now", "chat"))
        self.assertTrue(has_word("driv", "drive"))
        self.assertFalse(has_word("bright", "drive"))


class CommandLineTests(unittest.TestCase):
    def test_defaults_are_non_driving_and_use_default_models(self):
        args = parse_args([])
        self.assertIsNone(args.serial_port)
        self.assertIsNone(args.mic_device)
        self.assertFalse(args.offline)
        self.assertEqual(args.asr_model, "openai/whisper-tiny")

    def test_accepts_hardware_and_model_options(self):
        args = parse_args(
            [
                "--serial-port",
                "COM3",
                "--mic-device",
                "2",
                "--asr-model",
                "custom/asr",
                "--chat-model",
                "custom/chat",
                "--offline",
            ]
        )
        self.assertEqual(args.serial_port, "COM3")
        self.assertEqual(args.mic_device, 2)
        self.assertEqual(args.asr_model, "custom/asr")
        self.assertEqual(args.chat_model, "custom/chat")
        self.assertTrue(args.offline)


if __name__ == "__main__":
    unittest.main()
