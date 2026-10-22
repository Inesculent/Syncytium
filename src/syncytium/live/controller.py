"""The controller of the live swarm (``docs/preregistration-demo.md``).

On top of what every controller does (:mod:`syncytium.live.control`), it routes requests as
H8b does (the two lowest live members answer alone and every live member fuses when their
final numbers differ), runs probes, and orchestrates regrowth: it picks the regrowth data that
can rebuild the loss (refusing neighbour bits for adjacent losses), tells each new device which
survivors hold which files, and admits the regrown child by hash.
"""

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from syncytium.live.control import Control
from syncytium.live.wire import Endpoint
from syncytium.shards import (
    MANIFEST_NAME,
    SCALES_NAME,
    Manifest,
    Redundancy,
    RegrowthManifest,
    ShardError,
    shard_name,
)
from syncytium.shards.regrow import check_losses, regrowth_manifest_name, regrowth_name
from syncytium.tasks import gsm8k


class Controller(Control):
    """Runs the swarm's control plane.

    Args:
        endpoint: The controller's endpoint (index :data:`~syncytium.live.device.CONTROLLER`).
        reference: A directory holding the shard set's manifest and regrowth sidecars.
        tokenizer: The model's tokenizer, to encode prompts and decode nothing else.
        log: The JSON-lines event log to write.
        max_new_tokens: Most answer tokens per request.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        reference: Path,
        tokenizer: Any,
        log: Path,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(endpoint, log)
        self.tokenizer = tokenizer
        self.manifest = Manifest.load(reference / MANIFEST_NAME)
        self.sidecars = {
            kind: RegrowthManifest.load(reference / regrowth_manifest_name(kind))
            for kind in Redundancy
            if (reference / regrowth_manifest_name(kind)).exists()
        }
        self.max_new_tokens = max_new_tokens

    def expected_sha256(self, index: int) -> str:
        """The manifest's hash of child ``index``'s shard."""
        return self.manifest.shard_sha256[index]

    def serve(
        self,
        prompt: str,
        force: dict[str, int] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer one request, routed as in H8b, retrying while members die under it.

        ``force``, if given, makes every live member fuse, and tells member ``force["victim"]``
        to crash after ``force["die_after"]`` answer tokens. ``tags`` are logged with the
        request.
        """
        ids = self.tokenizer.encode(prompt)
        start, record = time.monotonic(), None
        while record is None and (live := self.live()):
            record = self._route(ids, live, force)
        record = record or {"route": "failed", "members": [], "text": None}
        record |= {"seconds": time.monotonic() - start}
        record["prediction"] = gsm8k.extract(record["text"]) if record["text"] else None
        record |= tags or {}
        self.log("request", **record)
        return record

    def probe(self, prompts: Sequence[str]) -> dict[str, Any]:
        """Teacher-force ``prompts`` through every live member together.

        Returns the members at the start, the SHA-256 over every prompt's logits, whether every
        member computed the same logits, whether every prompt finished with every starting
        member, and per prompt the top-1 tokens, the last positions' logits and the members
        that finished it. A member dropped from the group leaves only that prompt.
        """
        start, results, consistent = self.live(), [], True
        for k, prompt in enumerate(prompts):
            live = self.live()
            info = self._request_info(self.tokenizer.encode(prompt), live)
            request = self._send_all(live, "probe", info)
            replies = self._collect(request, live, "probe")
            for m, reply in replies.items():
                for event in reply.info["events"]:
                    fields = {"round": event["round"], "member": event["member"], "by": m}
                    self.log("probe-" + event["event"], prompt=k, **fields)
            finished = {m: r for m, r in replies.items() if not r.info.get("dropped")}
            consistent &= len({r.info["sha256"] for r in finished.values()}) == 1
            first = finished[min(finished)].info if finished else {"top1": [], "sha256": ""}
            results.append(
                {
                    "top1": first["top1"],
                    "sha256": first["sha256"],
                    "tail": finished[min(finished)].tensor if finished else None,
                    "members": first.get("members", []),
                }
            )
        digest = hashlib.sha256("".join(r["sha256"] for r in results).encode()).hexdigest()
        complete = all(r["members"] == start for r in results)
        self.log("probe", members=start, sha256=digest, consistent=consistent, complete=complete)
        return {
            "members": start,
            "sha256": digest,
            "consistent": consistent,
            "complete": complete,
            "prompts": results,
        }

    def regrow(self, lost: Sequence[int]) -> dict[str, Any]:
        """Regrow the children of ``lost`` on the new devices registered under their indices.

        Raises:
            ShardError: If no regrowth data the set holds can rebuild this loss.
        """
        lost = sorted(lost)
        kind = self._regrowth_kind(lost)
        survivors = [k for k in self.live() if k not in lost]
        common = [MANIFEST_NAME, *(regrowth_manifest_name(k) for k in self.sidecars)]
        common += [SCALES_NAME] if self.manifest.scales_sha256 else []
        files = {k: [shard_name(k), regrowth_name(kind, k)] for k in survivors}
        files[survivors[0]] += common
        info = {
            "lost": lost,
            "kind": str(kind),
            "restore": [str(k) for k in self.sidecars if k is not kind],
            "files": {str(k): names for k, names in files.items()},
            "sources": {str(k): list(self.devices[k].address) for k in survivors},
            "common": common,
        }
        self.log("regrowing", lost=lost, kind=str(kind), survivors=survivors)
        requests = {k: self._command(k, "regrow", info) for k in lost}
        report = {"kind": str(kind), "survivors": survivors, "devices": {}}
        for k, request in requests.items():
            reply = self._reply(request, k, ("regrown",))
            fields = reply.info if reply is not None else {"sha256": None}
            fields["matches"] = fields["sha256"] == self.manifest.shard_sha256[k]
            self.log("regrown", device=k, **fields)
            report["devices"][k] = fields
        report["admitted"] = self.admit(lost)
        return report

    heal = regrow

    def _route(
        self, ids: list[int], live: list[int], force: dict[str, int] | None
    ) -> dict[str, Any] | None:
        """One attempt at a request; ``None`` if a member it needed died first."""
        if force is None and len(live) == 1:
            return self._answer("solo", ids, live, "alone")
        if force is None:
            pair = live[:2]
            request = self._send_all(pair, "solo", self._request_info(ids, pair))
            answers = self._collect(request, pair, "answer")
            if len(answers) < len(pair):
                return None
            first, second = (answers[k].info for k in pair)
            if gsm8k.is_correct(gsm8k.extract(first["text"]), gsm8k.extract(second["text"]) or ""):
                return _record("agreed", pair, first)
        return self._answer("fuse", ids, live, "fused", force)

    def _answer(
        self,
        kind: str,
        ids: list[int],
        members: list[int],
        route: str,
        force: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        info = self._request_info(ids, members)
        request = next(self._requests)
        for k in members:
            extra = {"die_after": force["die_after"]} if force and force["victim"] == k else {}
            self._command(k, kind, info | extra, request)
        answers = self._collect(request, members, "answer")
        # A member the group dropped answers for itself only; the group's answer is the rest's.
        answers = {k: a for k, a in answers.items() if not a.info.get("dropped")}
        if not answers:
            return None
        return _record(route, members, answers[min(answers)].info)

    def _request_info(self, ids: list[int], members: list[int]) -> dict[str, Any]:
        addresses = {str(k): list(self.devices[k].address) for k in members}
        return {
            "prompt": ids,
            "max_new_tokens": self.max_new_tokens,
            "members": members,
            "addresses": addresses,
        }

    def _regrowth_kind(self, lost: list[int]) -> Redundancy:
        """The first regrowth data that can rebuild ``lost``; each refusal is logged."""
        size = self.manifest.scheme.num_children
        for kind in sorted(self.sidecars, key=list(Redundancy).index):
            try:
                check_losses(lost, size, kind, self.sidecars[kind].tolerated)
            except ShardError as error:
                self.log("regrowth-refused", lost=lost, kind=str(kind), reason=str(error))
                continue
            return kind
        raise ShardError(f"no regrowth data held can rebuild children {lost}")


def _record(route: str, members: list[int], answer: dict[str, Any]) -> dict[str, Any]:
    keep = ("request", "text", "tokens", "peak", "events", "timing")
    record = {"route": route, "members": members} | {k: answer[k] for k in keep if k in answer}
    record["finished_with"] = answer.get("members", members)
    return record
