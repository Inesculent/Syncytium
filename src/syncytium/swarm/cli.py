"""Command line: ``syncytium-swarm create MODEL DIR`` and ``syncytium-swarm console DIR``.

``create`` splits a model into staggered children with regrowth data. ``console`` opens an
interactive session over them (type ``help``); commands can also be piped in, one per line.
"""

import argparse
import cmd
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from syncytium.models import load_model, split_parameters
from syncytium.quant import Scheme
from syncytium.shards import Redundancy, ShardError, write_children, write_regrowth
from syncytium.swarm.session import Exchange, Format, Reply, Swarm

# Until H8 calibrates it, a round number: escalate when the agent's mean entropy exceeds it.
DEFAULT_THRESHOLD = 0.6


def main(argv: Sequence[str] | None = None) -> None:
    """Create a swarm from a model, or query one interactively."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="split a model into a swarm of children")
    create.add_argument("model", help="Hugging Face model id or local path")
    create.add_argument("directory", type=Path, help="output directory")
    create.add_argument("--bits", type=int, default=3)
    create.add_argument("--group-size", type=int, default=64)
    create.add_argument("--children", type=int, default=4)
    create.add_argument("--phase", type=float, default=0.5)
    create.add_argument("--rotation-seed", type=int, default=0, help="-1 for no rotation")
    create.add_argument(
        "--redundancy", type=Redundancy, choices=list(Redundancy), default=Redundancy.NEIGHBOUR
    )
    create.add_argument("--tolerate", type=int, default=1, help="losses the regrowth survives")
    console = commands.add_parser("console", help="query a swarm interactively")
    console.add_argument("directory", type=Path, help="directory written by create")
    console.add_argument("--device", default="cpu")
    console.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    console.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == "create":
        _create(args)
    else:
        swarm = Swarm.open(args.directory, args.device, args.max_new_tokens)
        Console(swarm, args.threshold).cmdloop()


def _create(args: argparse.Namespace) -> None:
    seed = None if args.rotation_seed < 0 else args.rotation_seed
    scheme = Scheme(args.bits, args.group_size, args.children, args.phase, seed)
    model = load_model(args.model, device="cpu", dtype=torch.bfloat16)
    write_children(*split_parameters(model), scheme, args.model, args.directory)
    write_regrowth(args.directory, args.redundancy, args.tolerate)
    print(f"{scheme.num_children} members in {args.directory}, regrowth: {args.redundancy}")


class Console(cmd.Cmd):
    """Interactive commands over a :class:`~syncytium.swarm.session.Swarm`."""

    intro = "Syncytium swarm. Type help for commands."
    prompt = "swarm> "

    def __init__(self, swarm: Swarm, threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__()
        self.swarm, self.threshold, self.form = swarm, threshold, Format.MATH
        self.use_rawinput = sys.stdin.isatty()
        if not self.use_rawinput:
            self.intro, self.prompt = None, ""

    def do_status(self, _: str) -> None:
        """status: members, regrowth data, escalation threshold and prompt format."""
        alive = self.swarm.alive()
        lost = [k for k in range(self.swarm.size) if k not in alive]
        scheme = self.swarm.manifest.scheme
        model = self.swarm.manifest.model
        print(f"model {model}, {scheme.bits}-bit children, N = {scheme.num_children}")
        print(f"alive {alive}, lost {lost}, regrowth {self.swarm.redundancy}")
        print(f"threshold {self.threshold:g} nats, format {self.form}")
        if lost:
            print(f"regrow: {self.swarm.regrowable() or 'possible'}")

    def do_ask(self, question: str) -> None:
        """Ask QUESTION: the leader answers; if unsure, every live member fuses and answers."""
        self._run(lambda: _show_exchange(self.swarm.ask(question, self.threshold, form=self.form)))

    def do_solo(self, line: str) -> None:
        """Solo K QUESTION: member K answers alone."""
        index, _, question = line.partition(" ")
        self._run(lambda: _show(self.swarm.answer(question, [int(index)], self.form)))

    def do_fuse(self, line: str) -> None:
        """Fuse all|K,K,... QUESTION: the given members (or every live one) fuse and answer."""
        who, _, question = line.partition(" ")
        members = self.swarm.alive() if who == "all" else [int(k) for k in who.split(",")]
        self._run(lambda: _show(self.swarm.answer(question, members, self.form)))

    def do_kill(self, line: str) -> None:
        """Kill K: lose member K (its files move to lost/)."""
        self._run(lambda: self._kill(int(line)))

    def do_regrow(self, _: str) -> None:
        """regrow: rebuild every lost member from the survivors, checked against its hash."""
        self._run(self._regrow)

    def do_threshold(self, line: str) -> None:
        """Threshold X: escalate when the agent's mean entropy exceeds X nats."""
        self._run(lambda: setattr(self, "threshold", float(line)))

    def do_format(self, line: str) -> None:
        """Format math|raw: GSM8K's few-shot prompt, or the text as given."""
        self._run(lambda: setattr(self, "form", Format(line.strip())))

    def do_quit(self, _: str) -> bool:
        """quit: leave the console."""
        return True

    do_exit = do_quit

    def default(self, line: str) -> bool:
        """End of input (cmd.Cmd passes it as the line "EOF") quits; anything else is unknown."""
        if line == "EOF":
            return True
        print(f"unknown command: {line.split()[0]} (type help)")
        return False

    def emptyline(self) -> bool:
        """An empty line does nothing (cmd.Cmd would repeat the last command)."""
        return False

    def _kill(self, index: int) -> None:
        self.swarm.kill(index)
        print(f"member {index} lost; alive {self.swarm.alive()}")
        problem = self.swarm.regrowable()
        if problem:
            print(f"warning: {problem}")

    def _regrow(self) -> None:
        regrown = self.swarm.regrow()
        for path in regrown:
            print(f"regrew {path.name}: matches the manifest")
        if not regrown:
            print("no member is lost")

    def _run(self, action: Callable[[], object]) -> None:
        try:
            action()
        except (ShardError, ValueError) as error:
            print(f"error: {error}")


def _show(reply: Reply) -> None:
    who = f"agent {reply.members[0]}" if len(reply.members) == 1 else f"fused {list(reply.members)}"
    print(f"[{who}] entropy {reply.entropy:.2f} nats, {reply.seconds:.1f} s")
    print(f"  {reply.text}")


def _show_exchange(exchange: Exchange) -> None:
    _show(exchange.solo)
    if exchange.fused is None:
        print(f"  (sure enough: entropy at most {exchange.threshold:g}, or no one to fuse with)")
    else:
        print(f"  (unsure: entropy above {exchange.threshold:g}, escalating)")
        _show(exchange.fused)


if __name__ == "__main__":
    main()
