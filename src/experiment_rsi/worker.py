import json
import os
import select
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


class CandidateExecutionError(RuntimeError):
    pass


class WorkerInfrastructureError(RuntimeError):
    pass


class PolicyWorker:
    def __init__(self, service, config, seed):
        self.timeout = config["timeout_seconds"]
        self.startup_timeout = config["startup_timeout_seconds"]
        self.container_name = "experiment-rsi-" + uuid.uuid4().hex[:12]
        self.tempdir = tempfile.TemporaryDirectory(
            prefix=".experiment-rsi-policy-", dir=Path.cwd()
        )
        Path(self.tempdir.name).chmod(0o755)
        program_path = Path(self.tempdir.name) / "program.py"
        program_path.write_text(service["policy_source"], encoding="utf-8")
        program_path.chmod(0o444)
        runtime_path = Path(__file__).with_name("worker_runtime.py").resolve()
        command = [
            "docker", "run", "--rm", "-i", "--name", self.container_name,
            "--network", "none", "--read-only", "--user", "65534:65534",
            "--memory", "256m", "--cpus", "0.5", "--pids-limit", "32",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", "type=bind,src={},dst=/program.py,readonly".format(program_path.resolve()),
            "--mount", "type=bind,src={},dst=/worker.py,readonly".format(runtime_path),
            config["image"], "python", "-B", "/worker.py", "/program.py", str(seed),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except BaseException:
            self.tempdir.cleanup()
            raise
        self.stdout_buffer = bytearray()
        self.stderr_tail = bytearray()
        try:
            ready = self._read_message(self.startup_timeout)
            if ready != {"ready": True}:
                raise WorkerInfrastructureError("policy worker returned an invalid startup message")
        except TimeoutError as error:
            self._stop_container()
            self.tempdir.cleanup()
            raise WorkerInfrastructureError(
                "policy worker did not start within {} seconds".format(self.startup_timeout)
            ) from error
        except BaseException:
            self._stop_container()
            self.tempdir.cleanup()
            raise

    def _read_message(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            newline = self.stdout_buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self.stdout_buffer[:newline])
                del self.stdout_buffer[:newline + 1]
                if len(raw) > 1024 * 1024:
                    raise CandidateExecutionError("candidate worker response exceeds 1 MiB")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise CandidateExecutionError("candidate wrote invalid worker output") from error
                if not isinstance(value, dict):
                    raise CandidateExecutionError("candidate worker response must be a JSON object")
                return value

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], remaining
            )
            if not readable:
                raise TimeoutError
            for stream in readable:
                chunk = os.read(stream.fileno(), 4096)
                if stream is self.process.stdout:
                    if not chunk:
                        detail = self.stderr_tail.decode("utf-8", errors="replace").strip()
                        raise WorkerInfrastructureError(
                            "policy worker exited before replying{}".format(": " + detail if detail else "")
                        )
                    self.stdout_buffer.extend(chunk)
                    if len(self.stdout_buffer) > 1024 * 1024:
                        raise CandidateExecutionError("candidate worker response exceeds 1 MiB")
                elif chunk:
                    self.stderr_tail.extend(chunk)
                    if len(self.stderr_tail) > 8192:
                        del self.stderr_tail[:-8192]

    def step(self, view, state):
        request = json.dumps(
            {"view": view, "state": state}, ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"
        self.process.stdin.write(request)
        self.process.stdin.flush()
        try:
            response = self._read_message(self.timeout)
        except TimeoutError as error:
            self._stop_container()
            raise CandidateExecutionError(
                "candidate policy exceeded {} seconds".format(self.timeout)
            ) from error
        if "candidate_error" in response:
            detail = response.get("traceback")
            message = response["candidate_error"]
            raise CandidateExecutionError(message + ("\n" + detail if detail else ""))
        if set(response) != {"result"} or not isinstance(response["result"], dict):
            raise CandidateExecutionError("candidate worker returned an invalid step result")
        result = response["result"]
        state = result["state"]
        directive = {key: value for key, value in result.items() if key != "state"}
        return directive, state

    def _stop_container(self):
        if self.process.poll() is not None:
            return
        result = subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0 and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
            raise WorkerInfrastructureError(
                "could not stop policy container: {}".format(result.stderr.strip())
            )
        self.process.wait()

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._stop_container()
        finally:
            self.tempdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
