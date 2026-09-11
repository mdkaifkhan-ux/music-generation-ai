import argparse
from pathlib import Path
import numpy as np
import pretty_midi
import tensorflow as tf

DATA_DIR = Path("data/midi")
MODEL_PATH = Path("music_lstm.keras")

NOTE_MIN, NOTE_MAX = 21, 108
VOCAB = NOTE_MAX - NOTE_MIN + 1

def midi_to_sequence(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in midi.instruments:
        for n in instrument.notes:
            if NOTE_MIN <= n.pitch <= NOTE_MAX:
                notes.append((n.start, n.pitch))
    notes.sort()
    return [p - NOTE_MIN for _, p in notes]

def load_sequences():
    files = list(DATA_DIR.glob("*.mid")) + list(DATA_DIR.glob("*.midi"))
    sequences = [midi_to_sequence(f) for f in files]
    sequences = [s for s in sequences if len(s) >= 20]
    if not sequences:
        raise FileNotFoundError(
            "Put MIDI files in data/midi/ before training."
        )
    return sequences

def train(epochs=20, seq_len=40):
    sequences = load_sequences()
    X, y = [], []
    for seq in sequences:
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i + seq_len])
            y.append(seq[i + seq_len])
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,)),
        tf.keras.layers.Embedding(VOCAB, 64),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(VOCAB, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.1)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

def generate(length=200, temperature=0.8):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first.")
    sequences = load_sequences()
    seed = sequences[0][:40]
    model = tf.keras.models.load_model(MODEL_PATH)
    result = list(seed)

    for _ in range(length):
        x = np.array([result[-40:]], dtype=np.int32)
        probs = model.predict(x, verbose=0)[0]
        logits = np.log(np.maximum(probs, 1e-8)) / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        result.append(int(np.random.choice(VOCAB, p=probs)))

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    for i, value in enumerate(result):
        pitch = value + NOTE_MIN
        start = i * 0.25
        piano.notes.append(
            pretty_midi.Note(
                velocity=90, pitch=pitch,
                start=start, end=start + 0.23
            )
        )
    midi.instruments.append(piano)
    out = Path("generated/ai_generated_music.mid")
    midi.write(str(out))
    print(f"Generated: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "generate"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    if args.action == "train":
        train(args.epochs)
    else:
        generate(args.length)
