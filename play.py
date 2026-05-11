"""Interactive keyboard player for the MAZE_SIMPLE renderer."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import tkinter as tk

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jes import (
    ACTION_INTERACT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    RayMazeEnv,
    State,
)
from jes.maps import (
    MAP_EPISODE_HORIZONS_BY_NAME,
    MAP_RENDER_KWARGS_BY_NAME,
    MAPS_BY_NAME,
)


KEY_ACTIONS = {
    "w": ACTION_MOVE_FORWARD,
    "s": ACTION_MOVE_BACKWARD,
    "a": ACTION_TURN_LEFT,
    "d": ACTION_TURN_RIGHT,
    "space": ACTION_INTERACT,
}
DEFAULT_TICK_MS = 50
DEFAULT_RECORD_PATH = "trajectory.gif"
DEFAULT_FRAME_PDF_STEM = "jaxenstein-frame"


class Player:
    def __init__(
        self,
        maze_name: str,
        scale: int,
        tick_ms: int,
        record_path: Path | None,
        render_size: tuple[int, int],
    ):
        render_w, render_h = render_size
        self.env = RayMazeEnv.from_ascii(
            [MAPS_BY_NAME[maze_name]],
            episode_horizons=MAP_EPISODE_HORIZONS_BY_NAME.get(maze_name),
            **MAP_RENDER_KWARGS_BY_NAME.get(maze_name, {}),
            img_h=render_h,
            img_w=render_w,
        )
        self.scale = scale
        self.tick_ms = tick_ms
        self.record_path = record_path
        self.recorded_frames: list[np.ndarray] = []
        self.frame_pdf_index = 0
        self.key = jax.random.key(0)
        self.step_fn = jax.jit(self.env.step)
        self.reset_fn = jax.jit(self.env.reset)

        self.obs, self.state = self._reset_env()
        self._warmup()
        self.pressed_keys: set[str] = set()
        self.running = True
        self.episode_done_announced = False

        self.root = tk.Tk()
        self.root.title("Jaxenstein")
        self.root.resizable(False, False)
        self.image_label = tk.Label(self.root, borderwidth=0, highlightthickness=0)
        self.image_label.pack()
        self.photo: tk.PhotoImage | None = None
        self.base_photo: tk.PhotoImage | None = None

        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)
        self.root.bind_all("q", self.close)
        self.root.bind_all("Q", self.close)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.draw()
        self.root.focus_force()
        self.root.after(self.tick_ms, self.tick)

    def _warmup(self) -> None:
        state = self.state
        for action in KEY_ACTIONS.values():
            _, state, _, _, _ = self.step_fn(state, jnp.asarray(action, dtype=jnp.int32))

    def on_key_press(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ("escape", "q"):
            self.close()
            return
        if key == "r":
            self.reset()
            return
        # P saves a one-off PDF snapshot without joining the held-action loop.
        if key == "p":
            if key not in self.pressed_keys:
                self.pressed_keys.add(key)
                self.save_frame_pdf()
            return
        if key not in KEY_ACTIONS:
            return

        self.pressed_keys.add(key)

    def on_key_release(self, event: tk.Event) -> None:
        self.pressed_keys.discard(event.keysym.lower())

    def _reset_env(self) -> tuple[jax.Array, State]:
        self.key, reset_key = jax.random.split(self.key)
        return self.reset_fn(reset_key, jnp.asarray(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.pressed_keys.clear()
        self.episode_done_announced = False
        self.obs, self.state = self._reset_env()
        self.draw()

    def tick(self) -> None:
        if not self.running:
            return

        actions = self._held_actions()
        if actions:
            for action in actions:
                self._step_action(action)
            self.draw()

        self.root.after(self.tick_ms, self.tick)

    def _held_actions(self) -> list[int]:
        actions = []
        turn_left = "a" in self.pressed_keys
        turn_right = "d" in self.pressed_keys
        move_forward = "w" in self.pressed_keys
        move_backward = "s" in self.pressed_keys
        interact = "space" in self.pressed_keys

        if turn_left != turn_right:
            actions.append(ACTION_TURN_LEFT if turn_left else ACTION_TURN_RIGHT)
        if move_forward != move_backward:
            actions.append(ACTION_MOVE_FORWARD if move_forward else ACTION_MOVE_BACKWARD)
        if interact:
            actions.append(ACTION_INTERACT)
        return actions

    def _step_action(self, action: int) -> None:
        self.obs, self.state, reward, done, _ = self.step_fn(
            self.state, jnp.asarray(action, dtype=jnp.int32)
        )
        if self.episode_done_announced:
            return
        if bool(reward):
            print("Goal reached. Press R to restart, or Q/Escape to quit.")
            self.episode_done_announced = True
        elif bool(done):
            print("Episode done. Press R to restart, or Q/Escape to quit.")
            self.episode_done_announced = True

    def draw(self) -> None:
        rgb = np.asarray(self.obs)
        if self.record_path is not None:
            self.recorded_frames.append(np.array(rgb, copy=True))

        height, width, _ = rgb.shape
        if self.base_photo is None:
            self.base_photo = tk.PhotoImage(width=width, height=height)

        rows = []
        for row in rgb:
            rows.append(
                "{"
                + " ".join(f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in row)
                + "}"
            )
        self.base_photo.put(" ".join(rows), to=(0, 0, width, height))
        self.photo = self.base_photo.zoom(self.scale, self.scale)
        self.image_label.configure(image=self.photo)

    def run(self) -> None:
        self.root.mainloop()

    def save_frame_pdf(self) -> None:
        path = self._next_frame_pdf_path()
        rgb = np.asarray(self.obs)
        frame = Image.fromarray(rgb).resize(
            (rgb.shape[1] * self.scale, rgb.shape[0] * self.scale),
            Image.Resampling.NEAREST,
        )
        frame.convert("RGB").save(path, "PDF")
        print(f"Saved frame to {path}")

    def _next_frame_pdf_path(self) -> Path:
        while True:
            self.frame_pdf_index += 1
            path = Path(f"{DEFAULT_FRAME_PDF_STEM}-{self.frame_pdf_index:04d}.pdf")
            if not path.exists():
                return path

    def close(self, event: tk.Event | None = None) -> None:
        del event
        if not self.running:
            return
        self.running = False
        self.pressed_keys.clear()
        self.save_recording()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def save_recording(self) -> None:
        if self.record_path is None or not self.recorded_frames:
            return

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [
            Image.fromarray(frame).resize(
                (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                Image.Resampling.NEAREST,
            )
            for frame in self.recorded_frames
        ]
        frames[0].save(
            self.record_path,
            save_all=True,
            append_images=frames[1:],
            duration=self.tick_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved recording to {self.record_path}")


def _parse_resolution(value: str) -> tuple[int, int]:
    raw = value.lower().replace(" ", "")
    if "x" in raw:
        width_text, height_text = raw.split("x", 1)
    else:
        width_text = height_text = raw

    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--resolution must be an integer or WIDTHxHEIGHT"
        ) from exc

    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("--resolution dimensions must be at least 1")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Integer display scale applied after native rendering.",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(64, 64),
        metavar="N|WIDTHxHEIGHT",
        help="Native render resolution. Use N for square output, e.g. 128, or WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=DEFAULT_TICK_MS,
        help="Milliseconds between repeated held-key updates.",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=DEFAULT_RECORD_PATH,
        default=None,
        metavar="PATH",
        help=f"Save the played trajectory as a GIF. Defaults to {DEFAULT_RECORD_PATH}.",
    )
    parser.add_argument(
        "--maze",
        choices=sorted(MAPS_BY_NAME),
        default="simple",
        help="Maze to play.",
    )
    args = parser.parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")
    if args.tick_ms < 1:
        raise ValueError("--tick-ms must be at least 1")

    print(
        "Controls: hold W/S to move, hold A/D to turn, Space interact, "
        "R reset, Q/Escape quit."
    )
    Player(
        args.maze,
        args.scale,
        args.tick_ms,
        None if args.record is None else Path(args.record),
        args.resolution,
    ).run()


if __name__ == "__main__":
    main()
