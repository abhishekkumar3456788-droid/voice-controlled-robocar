"""Voice-controlled assistant for a small robot car.

The speech and command-parsing helpers are safe to import without connecting to
audio hardware or loading machine-learning models. Run this file to start the
assistant.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from typing import Any

SAMPLE_RATE = 16_000
MIN_LEVEL = 0.04
MAX_LEVEL = 0.20
END_SILENCE_BLOCKS = 6
ASR_MODEL = "openai/whisper-tiny"
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

WORDS = {
    "forward": "F",
    "go": "F",
    "ahead": "F",
    "straight": "F",
    "back": "B",
    "backward": "B",
    "backwards": "B",
    "reverse": "B",
    "left": "L",
    "lift": "L",
    "loft": "L",
    "lef": "L",
    "right": "R",
    "write": "R",
    "rite": "R",
    "alright": "R",
    "stop": "S",
    "stoop": "S",
    "stob": "S",
    "step": "S",
}
FILLER = {"turn", "move", "to", "the", "please", "a", "now", "do", "it"}
SPEECH = {
    "F": "Going forward",
    "B": "Going back",
    "L": "Turning left",
    "R": "Turning right",
    "S": "Stopping",
}
HALLUCINATIONS = {
    "you",
    "thank you",
    "thanks",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "bye",
    "see you in the next video",
    "okay",
    "oh",
    "yeah",
}


def clean_words(text: str) -> list[str]:
    """Normalize recognized speech into lowercase words."""
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def is_hallucination(text: str) -> bool:
    """Identify common speech-recognition noise and empty transcriptions."""
    words = clean_words(text)
    return not words or " ".join(words) in HALLUCINATIONS


def to_command(text: str) -> str | None:
    """Return a car command for an unambiguous short utterance."""
    words = [word for word in clean_words(text) if word not in FILLER]
    if not words or len(words) > 3:
        return None

    found = {WORDS[word] for word in words if word in WORDS}
    if "S" in found:
        return "S"
    if len(found) == 1:
        return found.pop()
    return None


def has_word(text: str, target: str) -> bool:
    """Check for an exact or close-enough match to a mode-switch word."""
    for word in clean_words(text):
        if word == target:
            return True
        if len(word) >= 4 and difflib.get_close_matches(
            word, [target], n=1, cutoff=0.8
        ):
            return True
    return False


class VoiceCarAssistant:
    """Manage microphone input, speech models, optional serial control, and chat."""

    def __init__(
        self,
        serial_port: str | None = None,
        mic_device: int | None = None,
        asr_model: str = ASR_MODEL,
        chat_model: str = CHAT_MODEL,
        offline: bool = False,
    ) -> None:
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        import numpy as np
        import sounddevice as sd
        from transformers import pipeline

        self.np: Any = np
        self.sd: Any = sd
        self.mic_device = mic_device
        self.serial: Any = None
        self.history = [
            {
                "role": "system",
                "content": (
                    "You are a friendly voice assistant inside a small robot car. "
                    "Answer in one or two short sentences."
                ),
            }
        ]

        print("Loading speech-recognition model...")
        self.asr = pipeline("automatic-speech-recognition", model=asr_model)
        print("Loading chat model...")
        self.chat = pipeline("text-generation", model=chat_model)

        if serial_port:
            import serial

            self.serial = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(2)

    def send(self, code: str) -> None:
        print(
            f"  -> sending {code!r} to car"
            + ("" if self.serial else "  (no serial port set)")
        )
        if self.serial:
            self.serial.write(code.encode("ascii"))

    def speak(self, text: str) -> None:
        print(f"Car says: {text}")
        try:
            if sys.platform.startswith("win"):
                env = dict(os.environ, SPEAK_TEXT=text)
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object "
                        "System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Speak($env:SPEAK_TEXT)",
                    ],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                import pyttsx3

                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except (ImportError, OSError, subprocess.SubprocessError, RuntimeError) as error:
            print(f"Speech output unavailable: {error}")

    def calibrate(self) -> float:
        """Measure room noise so the trigger level adapts to the microphone."""
        print("Calibrating room noise, stay quiet for 2 seconds...")
        block = int(SAMPLE_RATE * 0.1)
        peaks = []
        with self.sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=self.mic_device,
        ) as stream:
            for index in range(25):
                data, _ = stream.read(block)
                if index >= 5:
                    peaks.append(self.np.abs(data).max())

        typical = float(self.np.median(peaks))
        level = min(MAX_LEVEL, max(MIN_LEVEL, typical * 4))
        print(
            f"Typical room noise {typical:.3f} (loudest {max(peaks):.3f}) "
            f"-> trigger level {level:.3f}"
        )
        return level

    def record_utterance(self, start_level: float, max_blocks: int) -> Any:
        block = int(SAMPLE_RATE * 0.1)
        pre, frames, started, silence, loud = [], [], False, 0, 0
        with self.sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=self.mic_device,
        ) as stream:
            while True:
                data, _ = stream.read(block)
                data = data.flatten()
                level = self.np.abs(data).max()
                if not started:
                    pre.append(data)
                    pre = pre[-4:]
                    if level >= start_level:
                        started = True
                        loud = 1
                        frames = pre[:]
                else:
                    frames.append(data)
                    if level >= start_level:
                        loud += 1
                    silence = silence + 1 if level < start_level * 0.6 else 0
                    if silence >= END_SILENCE_BLOCKS or len(frames) >= max_blocks:
                        break

        if len(frames) < 6 or loud < 2:
            return None
        padding = self.np.zeros(int(SAMPLE_RATE * 0.3), dtype="float32")
        return self.np.concatenate([padding, *frames, padding])

    def transcribe(self, wave: Any, max_tokens: int) -> str:
        generation = {
            "language": "en",
            "task": "transcribe",
            "max_new_tokens": max_tokens,
            "num_beams": 1,
        }
        result = self.asr(
            {"raw": wave, "sampling_rate": SAMPLE_RATE},
            generate_kwargs=generation,
        )
        return result["text"].strip()

    def reply_to(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        output = self.chat(
            self.history,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
        )
        answer = output[0]["generated_text"][-1]["content"].strip()
        self.history.append({"role": "assistant", "content": answer})
        del self.history[1:-8]
        return answer

    def run(self) -> None:
        start_level = self.calibrate()
        mode = "drive"
        self.speak("Ready. Say a command, or say chat to talk with me.")

        while True:
            print(f"[{mode} mode] Listening...")
            if mode == "drive":
                wave = self.record_utterance(start_level, max_blocks=40)
                if wave is None:
                    continue
                text = self.transcribe(wave, 10)
                if is_hallucination(text):
                    continue
                if has_word(text, "chat") and len(text.split()) <= 3:
                    self.send("S")
                    mode = "chat"
                    self.speak("Chat mode. Ask me anything. Say drive to go back.")
                    continue
                command = to_command(text)
                if command:
                    print(f"Heard: {text!r} -> EXECUTE {command}")
                    self.send(command)
                    self.speak(SPEECH[command])
                else:
                    print(f"Ignored: {text!r}")
            else:
                wave = self.record_utterance(start_level, max_blocks=100)
                if wave is None:
                    continue
                text = self.transcribe(wave, 60)
                if is_hallucination(text):
                    continue
                print(f"You said: {text!r}")
                if has_word(text, "drive") and len(text.split()) <= 3:
                    mode = "drive"
                    self.speak("Drive mode. Ready for commands.")
                    continue
                self.speak(self.reply_to(text))

    def close(self) -> None:
        try:
            self.send("S")
        finally:
            if self.serial:
                self.serial.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control a small robot car with voice commands."
    )
    parser.add_argument(
        "--serial-port",
        help="Arduino serial port (for example COM3); omit for console-only mode.",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        help="Microphone device index (default: system default microphone).",
    )
    parser.add_argument("--asr-model", default=ASR_MODEL, help="Hugging Face ASR model.")
    parser.add_argument(
        "--chat-model", default=CHAT_MODEL, help="Hugging Face text-generation model."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only models already cached locally.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assistant = VoiceCarAssistant(
        serial_port=args.serial_port,
        mic_device=args.mic_device,
        asr_model=args.asr_model,
        chat_model=args.chat_model,
        offline=args.offline,
    )
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
