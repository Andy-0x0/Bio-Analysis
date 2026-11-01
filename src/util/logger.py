import os
import re
import threading
import time
import weakref
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


class ABlockLogger:
    """
    The asynchronous block logger: every call renders its line up-front and queues it, and a worker thread
    mirrors the queue to stdout and to the files bound to the instance, sharing one handle per path across
    instances

    :param level:  the minimum level to log, one of DEBUG / INFO / WARNING / ERROR
    :param width:  the maximum width of a message line before it is wrapped; None disables wrapping
    """

    COLOR_RE: "re.Pattern[str]" = re.compile(r'\x1b\[[0-9;]*m')

    # Cross-instance stdout atomicity: all loggers share this lock around print()
    _stdout_lock: threading.Lock = threading.Lock()

    # Shared file entries {'fp', 'lock', 'refs', 'closed'} keyed by resolved absolute path, guarded by _registry_lock
    _file_registry: Dict[str, Dict[str, Any]] = {}
    _registry_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        width: Optional[int] = None
    ) -> None:
        """
        Build the logger and start its worker thread

        :param level:  the minimum level to log, one of DEBUG / INFO / WARNING / ERROR
        :param width:  the maximum width of a message line before it is wrapped; None disables wrapping

        :return None
        """
        self.level_map: Dict[str, int] = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
        }
        self.level: int = self.level_map[level.strip().upper()]
        self.width: Optional[int] = width

        self.role: Optional[str] = None
        self.last_time_stamp: Optional[str] = None
        self.queue: "Queue[Optional[str]]" = Queue()
        self.stop_event: threading.Event = threading.Event()

        # Space count for newline indentation alignment
        self.static_indent: Optional[int] = None

        # Serialises render + enqueue per instance, so time stamp de-duplication and line order hold across threads
        self._log_lock: threading.Lock = threading.Lock()

        # stdout toggle (default on); an Event so the worker can read it without holding a reference to self
        self._use_stdout: threading.Event = threading.Event()
        self._use_stdout.set()

        # Files bound by this instance as (key, entry); the entries are shared through the class-level registry
        self._files: List[Tuple[str, Dict[str, Any]]] = []
        self._files_lock: threading.Lock = threading.Lock()

        # The worker gets its state explicitly: holding self would keep the instance alive until interpreter exit
        self.worker: threading.Thread = threading.Thread(
            target=self._worker_loop,
            args=(self.queue, self.stop_event, self._use_stdout, self._files, self._files_lock),
            daemon=True,
        )
        self.worker.start()

        # Lifecycle cleanup
        self._finalizer: weakref.finalize = weakref.finalize(
            self,
            self._close,
            self.queue, self.stop_event, self.worker,
            self._files, self._files_lock,
        )

    @staticmethod
    def _worker_loop(
        queue: "Queue[Optional[str]]",
        stop_event: threading.Event,
        use_stdout: threading.Event,
        files: List[Tuple[str, Dict[str, Any]]],
        files_lock: threading.Lock,
    ) -> None:
        """
        Drain the queue on the worker thread, mirroring each line to stdout and to the bound files; static so the
        thread holds no self

        :param queue:       the line queue drained by the worker
        :param stop_event:  the event that tells the worker to stop
        :param use_stdout:  the event that is set while stdout output is enabled
        :param files:       the (key, entry) pairs bound by the instance
        :param files_lock:  the lock guarding files

        :return None
        """

        # Lines are written as they arrive; an idle tick (no line within 0.1s) flushes the file buffers
        while not stop_event.is_set() or not queue.empty():
            try:
                text = queue.get(timeout=0.1)
            except Empty:
                with files_lock:
                    files_snapshot = list(files)
                for _, entry in files_snapshot:
                    with entry['lock']:
                        if not entry['closed']:
                            try:
                                entry['fp'].flush()
                            except Exception:
                                pass
                continue

            if text is None:
                break

            # Nothing may kill the worker: a line that fails is dropped, and task_done() is guaranteed to run
            try:
                if use_stdout.is_set():
                    with ABlockLogger._stdout_lock:
                        try:
                            print(text)
                        except Exception:
                            pass

                with files_lock:
                    files_snapshot = list(files)

                if files_snapshot:
                    plain = ABlockLogger._de_color(str(text))
                    for _, entry in files_snapshot:
                        with entry['lock']:
                            if not entry['closed']:
                                try:
                                    entry['fp'].write(plain + '\n')
                                except Exception:
                                    pass
            except Exception:
                pass
            finally:
                queue.task_done()

    @property
    def _time_stamp(self) -> str:
        """
        The wall-clock stamp of the current line; a repeat within the same second collapses into blanks

        :return the stamp text, or spaces of the same width when it equals the previous stamp
        """
        curr_time_stamp = time.strftime("[%m/%d/%y %H:%M:%S]", time.localtime())
        if self.last_time_stamp is None or self.last_time_stamp != curr_time_stamp:
            self.last_time_stamp = curr_time_stamp
            return curr_time_stamp
        else:
            return ' ' * len(curr_time_stamp)

    @staticmethod
    def _close(
        queue: "Queue[Optional[str]]",
        stop_event: threading.Event,
        worker: threading.Thread,
        files: List[Tuple[str, Dict[str, Any]]],
        files_lock: threading.Lock,
    ) -> None:
        """
        Flush the queue, stop the worker and release the bound files; static so the finalizer holds no self

        :param queue:       the line queue drained by the worker
        :param stop_event:  the event that tells the worker to stop
        :param worker:      the worker thread
        :param files:       the (key, entry) pairs bound by the instance
        :param files_lock:  the lock guarding files

        :return None
        """

        # A worker that is dead (forked child) or is the current thread (finalizer fired on it) cannot be waited for
        draining = worker.is_alive() and threading.current_thread() is not worker
        if draining:
            queue.join()
        stop_event.set()
        queue.put(None)
        if draining:
            worker.join()

        with files_lock:
            local_files = list(files)
            files.clear()

        with ABlockLogger._registry_lock:
            for key, entry in local_files:
                if key in ABlockLogger._file_registry:
                    entry['refs'] -= 1
                    if entry['refs'] <= 0:
                        del ABlockLogger._file_registry[key]
                        with entry['lock']:
                            entry['closed'] = True
                            try:
                                entry['fp'].close()
                            except Exception:
                                pass

    @classmethod
    def _after_fork(cls) -> None:
        """
        Give a forked child fresh class-level state, since the inherited locks may be held by threads that do not
        exist there

        :return None
        """
        cls._stdout_lock = threading.Lock()
        cls._file_registry = {}
        cls._registry_lock = threading.Lock()

    @staticmethod
    def _role(role: str) -> str:
        """
        Wrap a role label in brackets

        :param role:  the role label

        :return the bracketed role label
        """
        return f"[{role}]"

    @staticmethod
    def _level(level: str) -> str:
        """
        Pad a level name to the fixed level column

        :param level:  the level name

        :return the level name left-aligned in 9 characters
        """
        return f"{level:9s}"

    def _do_print(self, level_digit: int) -> bool:
        """
        Check a level digit against the threshold; nothing is logged once the logger is closed

        :param level_digit:  the level digit as defined in level_map

        :return True if a line at that level should be logged
        """
        return level_digit >= self.level and not self.stop_event.is_set()

    @staticmethod
    def _red(text: str) -> str:
        """
        Colour a text red

        :param text:  the text to colour

        :return the text wrapped in ANSI red
        """
        return f"\033[91m{str(text)}\033[0m"

    @staticmethod
    def _green(text: str) -> str:
        """
        Colour a text green

        :param text:  the text to colour

        :return the text wrapped in ANSI green
        """
        return f"\033[92m{str(text)}\033[0m"

    @staticmethod
    def _yellow(text: str) -> str:
        """
        Colour a text yellow

        :param text:  the text to colour

        :return the text wrapped in ANSI yellow
        """
        return f"\033[93m{str(text)}\033[0m"

    @staticmethod
    def _blue(text: str) -> str:
        """
        Colour a text blue

        :param text:  the text to colour

        :return the text wrapped in ANSI blue
        """
        return f"\033[94m{str(text)}\033[0m"

    @staticmethod
    def _de_color(text: str) -> str:
        """
        Strip the ANSI colour codes from a text

        :param text:  the text to strip

        :return the text without colour codes
        """
        return ABlockLogger.COLOR_RE.sub('', text)

    def _format(self, role: str, message: str) -> str:
        """
        Indent continuation lines to the message column and wrap over-long lines at width

        :param role:     the role label of the line, used to compute the indent
        :param message:  the text to format

        :return the formatted message body
        """
        if self.static_indent is None:
            self.static_indent = len(str(self._time_stamp)) + 1 + 9 + 2

        indent = self.static_indent + len(self._role(role))
        padding = '\n' + (' ' * indent)

        # Fast path
        if (not self.width) or ((len(message) <= self.width) and ('\n' not in message)):
            return message if '\n' not in message else message.replace('\n', padding)

        # General path
        out: List[str] = []
        for line in message.split('\n'):
            line_len = len(line)
            if line_len <= self.width:
                out.append(line)
            else:
                out.extend(line[i:i + self.width] for i in range(0, line_len, self.width))

        return padding.join(out)

    def set_level(self, level: str) -> "ABlockLogger":
        """
        Set the minimum log level

        :param level:  the level name, one of DEBUG / INFO / WARNING / ERROR (case-insensitive)

        :return self, for call chaining
        """
        self.level = self.level_map[level.strip().upper()]
        return self

    def set_role(self, role: str) -> "ABlockLogger":
        """
        Set the default role label

        :param role:  the role label used when a call gives none

        :return self, for call chaining
        """
        self.role = role
        return self

    def bind_stdout(self) -> "ABlockLogger":
        """
        Enable stdout output (on by default)

        :return self, for call chaining
        """
        self._use_stdout.set()
        return self

    def unbind_stdout(self) -> "ABlockLogger":
        """
        Disable stdout output; lines already queued but not yet written skip stdout as well

        :return self, for call chaining
        """
        self._use_stdout.clear()
        return self

    def bind_file(self, path: Union[str, Path], mode: str = 'a') -> "ABlockLogger":
        """
        Mirror output to a file; loggers bound to the same path share one handle, its lock and its open mode

        :param path:  the path of the file; parent directories are created on demand
        :param mode:  the open mode, honoured only by the first logger to bind the path

        :return self, for call chaining
        """
        path = Path(path).resolve()
        key = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # The duplicate check and the append share one critical section, so concurrent binds cannot double-write
        with self._files_lock:
            for k, _ in self._files:
                if k == key:
                    return self

            with ABlockLogger._registry_lock:
                entry = ABlockLogger._file_registry.get(key)
                if entry is None:
                    entry = {
                        'fp': open(key, mode, encoding='utf-8'),
                        'lock': threading.Lock(),
                        'refs': 1,
                        'closed': False,
                    }
                    ABlockLogger._file_registry[key] = entry
                else:
                    entry['refs'] += 1

            self._files.append((key, entry))

        return self

    def unbind_files(self) -> "ABlockLogger":
        """
        Unbind all files from this instance, closing a shared handle once its last reference drops; lines already
        queued but not yet written skip the files as well

        :return self, for call chaining
        """
        with self._files_lock:
            local_files = list(self._files)
            self._files.clear()

        with ABlockLogger._registry_lock:
            for key, entry in local_files:
                if key in ABlockLogger._file_registry:
                    entry['refs'] -= 1
                    if entry['refs'] <= 0:
                        del ABlockLogger._file_registry[key]
                        with entry['lock']:
                            entry['closed'] = True
                            try:
                                entry['fp'].close()
                            except Exception:
                                pass

        return self

    def info(self, message: str, role: Optional[str] = None, raw: bool = False) -> str:
        """
        Log at INFO level; raw=True bypasses all formatting

        :param message:  the text to log
        :param role:     the role label; falls back to the default role, then to "Logger"
        :param raw:      when True the message is queued as-is: no timestamp, level, role, indent or reflow

        :return the text that was queued, or an empty string if the level is filtered out
        """
        if self._do_print(1):
            with self._log_lock:
                if raw:
                    text = message
                else:
                    role = role or self.role or "Logger"
                    text = f"{self._time_stamp} {self._blue(self._level('INFO'))}{self._role(role)}: {self._format(role, message)}"
                self.queue.put(text)
            return text
        else:
            return ''

    def debug(self, message: str, role: Optional[str] = None, raw: bool = False) -> str:
        """
        Log at DEBUG level; raw=True bypasses all formatting

        :param message:  the text to log
        :param role:     the role label; falls back to the default role, then to "Logger"
        :param raw:      when True the message is queued as-is: no timestamp, level, role, indent or reflow

        :return the text that was queued, or an empty string if the level is filtered out
        """
        if self._do_print(0):
            with self._log_lock:
                if raw:
                    text = message
                else:
                    role = role or self.role or "Logger"
                    text = f"{self._time_stamp} {self._green(self._level('DEBUG'))}{self._role(role)}: {self._format(role, message)}"
                self.queue.put(text)
            return text
        else:
            return ''

    def warning(self, message: str, role: Optional[str] = None, raw: bool = False) -> str:
        """
        Log at WARNING level; raw=True bypasses all formatting

        :param message:  the text to log
        :param role:     the role label; falls back to the default role, then to "Logger"
        :param raw:      when True the message is queued as-is: no timestamp, level, role, indent or reflow

        :return the text that was queued, or an empty string if the level is filtered out
        """
        if self._do_print(2):
            with self._log_lock:
                if raw:
                    text = message
                else:
                    role = role or self.role or "Logger"
                    text = f"{self._time_stamp} {self._yellow(self._level('WARNING'))}{self._role(role)}: {self._format(role, message)}"
                self.queue.put(text)
            return text
        else:
            return ''

    def error(self, message: str, role: Optional[str] = None, raw: bool = False) -> str:
        """
        Log at ERROR level; raw=True bypasses all formatting

        :param message:  the text to log
        :param role:     the role label; falls back to the default role, then to "Logger"
        :param raw:      when True the message is queued as-is: no timestamp, level, role, indent or reflow

        :return the text that was queued, or an empty string if the level is filtered out
        """
        if self._do_print(3):
            with self._log_lock:
                if raw:
                    text = message
                else:
                    role = role or self.role or "Logger"
                    text = f"{self._time_stamp} {self._red(self._level('ERROR'))}{self._role(role)}: {self._format(role, message)}"
                self.queue.put(text)
            return text
        else:
            return ''

    def close(self) -> None:
        """
        Flush the queue, stop the worker thread and release the bound files; safe to call more than once

        :return None
        """
        self._finalizer()


# A forked child inherits the class-level locks in whatever state the parent's threads left them
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=ABlockLogger._after_fork)