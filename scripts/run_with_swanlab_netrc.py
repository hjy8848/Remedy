import netrc
import os
from pathlib import Path
import sys


credentials = netrc.netrc(str(Path.home() / ".swanlab/.netrc"))
auth = credentials.authenticators("https://api.swanlab.cn")
if not auth or not auth[2]:
    raise RuntimeError("SwanLab API key is missing from ~/.swanlab/.netrc")

os.environ["SWANLAB_API_KEY"] = auth[2]
os.execv(
    sys.executable,
    [sys.executable, "-c", "from experiment_rsi.cli import main; main()", *sys.argv[1:]],
)
