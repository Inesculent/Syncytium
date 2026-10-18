"""GSM8K with the standard 8-shot chain-of-thought prompt (``docs/preregistration-swarm.md``).

The exemplars are the eight of Wei et al. (2022), in the "Q: ... A: ..." format used by most
evaluations of base models. Generation stops at a blank line or the next "Q:", and the last
number in the answer is compared with the reference numerically.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from datasets import load_dataset

GSM8K = ("openai/gsm8k", "main")
STOPS = ("\n\n", "Q:")

EXEMPLARS = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers "
        "plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in "
        "the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they "
        "have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
        "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How "
        "many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave "
        "Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How "
        "many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 "
        "more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each "
        "day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 "
        "more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she "
        "has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)

_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


@dataclass(frozen=True)
class Problem:
    """One GSM8K problem.

    Attributes:
        question: The problem statement.
        answer: The reference answer, a number as written after "####".
    """

    question: str
    answer: str


def load_gsm8k(split: str, limit: int | None = None) -> list[Problem]:
    """Problems of a GSM8K split, in dataset order; ``limit`` keeps the first few."""
    rows = load_dataset(*GSM8K, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [Problem(row["question"], reference(row["answer"])) for row in rows]


def reference(solution: str) -> str:
    """The final answer of a GSM8K solution, the text after "####"."""
    return solution.rsplit("####", 1)[-1].strip()


def prompt(question: str) -> str:
    """The 8-shot prompt for ``question``; the model continues after "A:"."""
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in EXEMPLARS)
    return f"{shots}Q: {question}\nA:"


def truncate(text: str) -> str:
    """Cut a generated answer at the first stop sequence."""
    cut = min((i for i in (text.find(stop) for stop in STOPS) if i >= 0), default=len(text))
    return text[:cut]


def extract(answer: str) -> str | None:
    """The last number in an answer, without thousands separators, or ``None``."""
    numbers = _NUMBER.findall(answer)
    return numbers[-1].replace(",", "") if numbers else None


def is_correct(prediction: str | None, gold: str) -> bool:
    """Whether ``prediction`` equals ``gold`` as a number (so 18, 18.0 and 18.00 agree)."""
    if prediction is None:
        return False
    try:
        return Decimal(prediction) == Decimal(gold.replace(",", ""))
    except InvalidOperation:
        return False
