#!/usr/bin/env python3
"""CLI entry point.

    python main.py script.json > output.wav
    python main.py script.json -o output.wav      (equivalent, no shell redirection needed)

Renders a JSON script with Vocalizer and writes the resulting wav audio as raw
bytes to stdout (unless -o is given). All logging/progress text goes to
stderr, so stdout stays clean for the ">" redirection to work.
"""

import argparse
import io
import json
import sys

import soundfile as sf

from vocalizer import Vocalizer, VocalizerConfig


def main():
    parser = argparse.ArgumentParser(description="Render a vocalizer JSON script to wav audio.")
    parser.add_argument("script_path", help="Path to the JSON script file")
    parser.add_argument("-o", "--output", help="Write to this file instead of stdout")
    parser.add_argument("--sounds-dir", default="./sounds", help="Sound library directory (default: ./sounds)")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Output sample rate (default: 48000)")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Default VoxCPM cfg_value (default: 2.0)")
    parser.add_argument("--inference-timesteps", type=int, default=10,
                         help="Default VoxCPM inference_timesteps (default: 10)")
    parser.add_argument("--target-loudness-db", type=float, default=-20.0,
                         help="RMS loudness normalization target in dBFS (default: -20.0)")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2", help="VoxCPM model id or local path")
    parser.add_argument("--denoise", action="store_true", help="Enable VoxCPM's ZipEnhancer denoiser on load")
    args = parser.parse_args()

    with open(args.script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    config = VocalizerConfig(
        sound_library_dir=args.sounds_dir,
        output_sample_rate=args.sample_rate,
        voxcpm_model_id=args.model_id,
        load_denoiser=args.denoise,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        target_loudness_db=args.target_loudness_db,
    )

    print(f"Loading {args.model_id}...", file=sys.stderr)
    vocalizer = Vocalizer(config)

    print(f"Rendering {args.script_path}...", file=sys.stderr)
    audio = vocalizer.render_script(script)

    if args.output:
        sf.write(args.output, audio, config.output_sample_rate)
    else:
        buffer = io.BytesIO()
        sf.write(buffer, audio, config.output_sample_rate, format="WAV")
        sys.stdout.buffer.write(buffer.getvalue())
        sys.stdout.buffer.flush()

    print(f"Done: {audio.shape[0] / config.output_sample_rate:.2f}s rendered", file=sys.stderr)


if __name__ == "__main__":
    main()
