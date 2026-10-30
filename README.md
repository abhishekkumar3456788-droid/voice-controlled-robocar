# Voice-Controlled Robot Car

A Python voice assistant for a small robot car. It listens for short driving
commands, sends single-character instructions over an optional Arduino serial
connection, and has a separate conversational mode.

## Features

- Speech recognition with Whisper (`openai/whisper-tiny` by default).
- Short, explicit driving commands with conservative handling of ambiguous input.
- Automatic microphone noise calibration and silence-based recording.
- Optional Arduino control over serial at 9600 baud.
- Optional chat mode powered by a small instruction-tuned language model.
- Safe to import for testing; audio devices and models are initialized only when
  the program is started.

## Safety first

This is a hobby project, not a safety-certified vehicle controller. Test with the
car's wheels lifted off the ground, use a physical power switch, keep a clear
area around the car, and never rely on speech recognition as an emergency stop.
The program sends `S` when chat mode begins and during a normal shutdown, but a
power loss, process crash, or serial disconnect may prevent that command from
reaching the car. Configure the motor controller to stop on its own if it stops
receiving commands.

## Requirements

- Python 3.10 or newer
- A working microphone
- A compatible audio backend for `sounddevice`
- Internet access the first time each Hugging Face model is loaded, unless the
  models have already been cached
- Optional Arduino-compatible board and firmware that accepts the protocol below

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyserial       # only if controlling a car over serial
```

On Linux, installing `sounddevice` may also require the system PortAudio library.
On Windows, the default speech output uses Windows Speech; on macOS and Linux,
install `pyttsx3` if you want spoken replies:

```powershell
python -m pip install pyttsx3
```

The default Whisper and chat models are downloaded and cached on first use. Both
models can require substantial memory and disk space. To run without network
access, first load both models successfully while online, then use `--offline`.

## Run

Start in console-only mode (no car commands are sent):

```powershell
python car_project.py
```

Connect an Arduino on a serial port:

```powershell
python car_project.py --serial-port COM3
```

Choose a microphone by device index if the system default is not the one you
want:

```powershell
python car_project.py --mic-device 2 --serial-port COM3
```

List audio devices with this one-liner:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Use locally cached models only:

```powershell
python car_project.py --offline
```

Say **forward**, **back**, **left**, **right**, or **stop** to drive. Say **chat**
to enter conversational mode, and **drive** to return to driving. Stop the
program with `Ctrl+C`.

## Arduino serial protocol

The application opens the configured port at **9600 baud** and sends one ASCII
character per recognized command:

| Character | Meaning |
| --- | --- |
| `F` | Forward |
| `B` | Backward |
| `L` | Left |
| `R` | Right |
| `S` | Stop |

This repository does not include car-specific firmware. Ensure your board's
firmware uses the same protocol and has an independent motor timeout or other
fail-safe.

## Tests

The unit tests cover command parsing and do not require a microphone, model
downloads, or serial hardware:

```powershell
python -m unittest discover -s tests -v
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
