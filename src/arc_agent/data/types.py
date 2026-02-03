"""Type definitions for ARC tasks."""

from typing import List
from pydantic import BaseModel, Field


# Grid is a 2D array of integers 0-9 (colors), size 1x1 to 30x30
Grid = List[List[int]]


class Example(BaseModel):
    """A single input/output example pair."""
    input: Grid
    output: Grid


class Task(BaseModel):
    """An ARC task with training examples and test cases."""
    task_id: str = Field(description="Unique task identifier (filename without .json)")
    train: List[Example] = Field(description="Training demonstration pairs")
    test: List[Example] = Field(description="Test evaluation pairs")

    @property
    def num_train(self) -> int:
        return len(self.train)

    @property
    def num_test(self) -> int:
        return len(self.test)


# Color constants (0-9)
class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GRAY = 5
    MAGENTA = 6
    ORANGE = 7
    CYAN = 8
    MAROON = 9


COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}
