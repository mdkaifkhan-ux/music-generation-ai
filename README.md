# Task 3: Music Generation with AI

An LSTM-based neural network learns note sequences from MIDI files and
generates a new MIDI composition.

## Setup
```bash
pip install -r requirements.txt
```

Put `.mid` or `.midi` training files inside `data/midi/`.

## Train
```bash
python app.py train --epochs 20
```

## Generate
```bash
python app.py generate --length 200
```

The generated MIDI file is saved as `generated/ai_generated_music.mid`.
