import sys
import threading
import time

from syncytium.live import host
from syncytium.live.host import RUNNER, Host
from syncytium.live.wire import Endpoint, Message


def test_an_idle_host_keeps_its_devices_until_the_runner_is_gone(tmp_path, monkeypatch):
    # The first 4B run lost its devices when the hosts gave up after an hour without commands.
    monkeypatch.setattr(host, "IDLE", 0.2)
    runner = Endpoint(RUNNER, host="127.0.0.1")
    node = Host(100, runner.address, ["0"])
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    hello = runner.inbox.take(lambda m: m.kind == "host-hello", 5)
    address = tuple(hello.info["address"])
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    info = {"argv": sleeper, "gpu": "0", "log": str(tmp_path / "device.log"), "request": 1}
    runner.send(100, address, Message("spawn", RUNNER, info=info))
    assert runner.inbox.take(lambda m: m.kind == "done", 5) is not None

    time.sleep(1.0)  # five idle periods with the runner still there
    assert thread.is_alive() and node.processes[0].poll() is None

    runner.close()
    thread.join(timeout=10)
    assert not thread.is_alive()
    node.processes[0].wait(timeout=10)
    assert node.processes[0].returncode is not None
