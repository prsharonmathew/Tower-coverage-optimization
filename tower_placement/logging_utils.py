from __future__ import annotations

from pathlib import Path


class SearchLogger:
    def __init__(self, quiet: bool, log_file: Path) -> None:
        self.quiet = quiet
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)

    def debug(self, text: str) -> None:
        if not self.quiet:
            print(text)

    def info(self, text: str) -> None:
        print(text)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()

    def clear(self) -> None:
        self.log_file.write_text("", encoding="utf-8")
