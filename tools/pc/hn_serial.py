"""Helpers shared by the HushNet PC scripts.

Needs pyserial:  pip install pyserial
The firmware prints lines such as
    RX type=test node=1 burst=3 seq=7 len=20 rssi=-41.5 snr=9.75 ferr_hz=-812 lost=0 t_ms=123456
i.e. a tag followed by key=value pairs. parse_line() turns such a line into (tag, dict).
"""
import queue
import re
import sys
import threading
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")

_KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def parse_line(line):
    line = line.strip()
    if not line:
        return None, {}
    tag = line.split(None, 1)[0]
    return tag, {k: v.strip('"') for k, v in _KV.findall(line)}


class Board:
    """One board on a serial port. A background thread collects complete lines."""

    def __init__(self, port, name=None, baud=115200, echo=False, raw_log=None):
        self.port = port
        self.name = name or port
        self.echo = echo
        self.raw_log = raw_log
        self.q = queue.Queue()
        self._stop = False
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.1
        # Keep DTR and RTS low: on the native USB port of the ESP32 these lines can reset the chip.
        self.ser.dtr = False
        self.ser.rts = False
        try:
            self.ser.open()
        except serial.SerialException as e:
            sys.exit(f"Cannot open {port}: {e}\nClose the Arduino Serial Monitor (or any other program using the port) and try again.")
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                data = self.ser.read(256)
            except Exception:
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").rstrip("\r")
                ts = time.time()
                if self.raw_log:
                    self.raw_log.write(f"{ts:.3f}\t{self.name}\t{line}\n")
                    self.raw_log.flush()
                if self.echo:
                    print(f"[{self.name}] {line}")
                self.q.put((ts, line))

    def send(self, cmd):
        self.ser.write((cmd + "\n").encode())
        self.ser.flush()

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout):
        """Next (timestamp, line) or None after timeout seconds."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for(self, tags, timeout):
        """Reads lines until one starts with a tag in `tags`.
        Returns (tag, fields, lines_seen); tag is None on timeout."""
        seen = []
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                return None, {}, seen
            item = self.read(left)
            if item is None:
                return None, {}, seen
            tag, fields = parse_line(item[1])
            seen.append(item)
            if tag in tags:
                return tag, fields, seen

    def command(self, cmd, tags, timeout=5.0):
        self.drain()
        self.send(cmd)
        return self.wait_for(tags, timeout)

    def hello(self):
        """Checks that the HushNet firmware answers. Returns the fields of the first INFO line."""
        time.sleep(0.3)
        for _ in range(4):
            tag, fields, _seen = self.command("INFO", ("INFO",), timeout=3.0)
            if tag:
                # let the remaining INFO, CFG and STAT lines arrive
                time.sleep(0.3)
                self.drain()
                return fields
        sys.exit(f"No answer from the board on {self.port}. Is hushnet_poc flashed, and is the Serial Monitor closed?")

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except Exception:
            pass


def list_ports():
    from serial.tools import list_ports as lp
    rows = []
    for p in lp.comports():
        rows.append((p.device, p.description, f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else ""))
    return rows


if __name__ == "__main__":
    # python hn_serial.py   -> list the serial ports
    for dev, desc, vidpid in list_ports():
        print(f"{dev:10s} {vidpid:10s} {desc}")
