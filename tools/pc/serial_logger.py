"""Serial monitor and CSV logger for one HushNet board.

    python serial_logger.py COM5                         show everything the board prints
    python serial_logger.py COM5 --csv rx_log.csv        also save RX, TX, PING, INFER, STAT lines as CSV
    python serial_logger.py COM5 --cmd "MODE RX"         send a command right after connecting (repeatable)
    python serial_logger.py --list                       list the serial ports

While it runs you can type commands (HELP, INFO, CFG sf=9, MODE RX, ...) and press Enter.
Stop with Ctrl+C. Close the Arduino Serial Monitor first: only one program can use a port.
"""
import argparse
import csv
import datetime
import sys
import threading
import time

from hn_serial import Board, list_ports, parse_line

CSV_TAGS = ("RX", "TX", "PING", "INFER", "INFER_SUMMARY", "STAT", "CFG")
CSV_FIELDS = ["pc_time", "tag", "type", "node", "burst", "seq", "class", "name", "conf", "len", "rssi", "snr", "ferr_hz", "lost",
              "toa_meas_us", "toa_calc_us", "t_ms", "hex", "line"]


def main():
    ap = argparse.ArgumentParser(description="Serial monitor and CSV logger for a HushNet board")
    ap.add_argument("port", nargs="?", help="serial port, for example COM5 or /dev/ttyACM0")
    ap.add_argument("--csv", help="CSV file to append to")
    ap.add_argument("--cmd", action="append", default=[], help="command to send after connecting (can be repeated)")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    a = ap.parse_args()
    if a.list or not a.port:
        for dev, desc, vidpid in list_ports():
            print(f"{dev:10s} {vidpid:10s} {desc}")
        return
    board = Board(a.port)
    writer, fh = None, None
    if a.csv:
        new = True
        try:
            new = open(a.csv).read(1) == ""
        except OSError:
            pass
        fh = open(a.csv, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()

    def stdin_loop():
        for line in sys.stdin:
            line = line.strip()
            if line:
                board.send(line)

    threading.Thread(target=stdin_loop, daemon=True).start()
    time.sleep(0.3)
    for c in a.cmd:
        board.send(c)
        time.sleep(0.2)
    print(f"connected to {a.port}. Type a command and press Enter, Ctrl+C to stop.")
    try:
        while True:
            item = board.read(0.5)
            if item is None:
                continue
            ts, line = item
            stamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
            print(f"{stamp} {line}")
            if writer:
                tag, fields = parse_line(line)
                if tag in CSV_TAGS:
                    row = dict(fields)
                    row.update(pc_time=datetime.datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"), tag=tag, line=line)
                    writer.writerow(row)
                    fh.flush()
    except KeyboardInterrupt:
        pass
    finally:
        board.close()
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
