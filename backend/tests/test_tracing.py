import subprocess
import sys


def test_stack_dumps_fire_while_block_is_slow_and_stop_after():
    # faulthandler writes to the raw stderr file descriptor, which pytest's
    # capture can't intercept -- run it in a child process and read its stderr.
    script = (
        "import time\n"
        "from app.core.tracing import stack_dumps_while_slow\n"
        "def slow_model_load():\n"
        "    time.sleep(2.5)\n"
        "with stack_dumps_while_slow('test load', every_s=1):\n"
        "    slow_model_load()\n"
        "time.sleep(1.5)  # no dumps should fire once the block has exited\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert "watching test load" in result.stderr
    dumps = result.stderr.count("slow_model_load")
    assert dumps == 2, result.stderr  # fired at ~1s and ~2s, not after the block ended
