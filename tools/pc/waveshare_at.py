"""Waveshare USB-TO-LoRa-HF (B) helper: configure it with AT commands and test whether it can
exchange packets with the HushNet boards.

The Waveshare wiki says this dongle "only supports" its own product family, and it does not
document the frame it puts on air. So interoperability with a raw LoRa packet from RadioLib is
not guaranteed; the "probe" command below finds out by experiment.

    python waveshare_at.py setup COM7              low power (10 dBm), SF7, 125 kHz, 4/5, channel 18 (868 MHz), stream mode
    python waveshare_at.py at COM7 AT+VER AT+HELP  send any AT commands (enters and leaves AT mode around them)
    python waveshare_at.py listen COM7             print whatever the dongle receives over the air
    python waveshare_at.py send COM7 "hello" --count 3
    python waveshare_at.py probe COM7 --board COM5 run the interoperability experiment with a HushNet board

IMPORTANT: the factory setting is 22 dBm transmit power. Run "setup" (it sets AT+PWR=10 first)
before you let the dongle transmit, and check in the printed replies that the command was accepted.

AT syntax used here is the one on https://www.waveshare.com/wiki/USB-TO-LoRa-xF :
enter AT mode with "+++" + CR LF, leave it with "AT+EXIT" + CR LF, 115200 baud 8N1.
The wiki does not document the reply texts, so this script simply prints what comes back.
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")

from hn_serial import Board, parse_line

BW_CODE = {125: 0, 250: 1, 500: 2}


def open_dongle(port):
    try:
        return serial.Serial(port, 115200, timeout=0.1)
    except serial.SerialException as e:
        sys.exit(f"Cannot open {port}: {e}")


def read_quiet(ser, first=0.6, quiet=0.25):
    """Reads until nothing has arrived for `quiet` seconds."""
    data = b""
    end = time.time() + first
    while time.time() < end:
        chunk = ser.read(256)
        if chunk:
            data += chunk
            end = max(end, time.time() + quiet)
    return data


def at(ser, cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\r\n").encode())
    reply = read_quiet(ser)
    text = reply.decode("utf-8", "replace").strip()
    print(f"  > {cmd:<16s} < {text if text else '(no reply)'}")
    return text


def at_session(ser, commands):
    at(ser, "+++")
    out = [at(ser, c) for c in commands]
    at(ser, "AT+EXIT")
    time.sleep(0.5)          # the wiki says new settings apply after leaving AT mode
    return out


def setup_commands(a):
    if a.bw not in BW_CODE:
        sys.exit("--bw must be 125, 250 or 500")
    return ["AT+VER", f"AT+PWR={a.pwr}", f"AT+SF={a.sf}", f"AT+BW={BW_CODE[a.bw]}", f"AT+CR={a.cr - 4}",
            f"AT+TXCH={a.ch}", f"AT+RXCH={a.ch}", "AT+MODE=1", "AT+LBT=0", "AT+RSSI=0"]


def hexdump(data):
    return data.hex(" ") + "   |" + "".join(chr(b) if 32 <= b < 127 else "." for b in data) + "|"


def cmd_setup(a):
    ser = open_dongle(a.port)
    print(f"configuring the dongle on {a.port}: {a.pwr} dBm, SF{a.sf}, {a.bw} kHz, 4/{a.cr}, channel {a.ch}, stream mode")
    at_session(ser, setup_commands(a))
    print("done. If a line shows an error or no reply, do not transmit before fixing it (see README, Waveshare section).")
    ser.close()


def cmd_at(a):
    ser = open_dongle(a.port)
    at_session(ser, a.commands)
    ser.close()


def cmd_listen(a):
    ser = open_dongle(a.port)
    print("listening, Ctrl+C to stop")
    try:
        while True:
            data = read_quiet(ser, first=0.2, quiet=0.15)
            if data:
                print(time.strftime("%H:%M:%S"), f"{len(data):3d} bytes  {hexdump(data)}")
    except KeyboardInterrupt:
        pass
    ser.close()


def cmd_send(a):
    ser = open_dongle(a.port)
    for i in range(a.count):
        ser.write(a.text.encode())
        ser.flush()
        print(f"sent {len(a.text)} bytes ({i + 1}/{a.count})")
        time.sleep(a.gap)
    ser.close()


def board_cfg(board, text):
    tag, f, seen = board.command(text, ("CFG", "ERR"), 5.0)
    if tag != "CFG":
        sys.exit(f"board: '{text}' failed: {seen[-1][1] if seen else 'no answer'}")
    return f


def collect_rx(board, seconds):
    """All RX lines of the board during `seconds`."""
    rows = []
    end = time.time() + seconds
    while time.time() < end:
        item = board.read(0.1)
        if item:
            tag, f = parse_line(item[1])
            if tag == "RX":
                rows.append(f)
    return rows


def cmd_probe(a):
    ser = open_dongle(a.port)
    board = Board(a.board, "board")
    info = board.hello()
    freq = 850.0 + a.ch          # the wiki gives channel 18 = 868 MHz and 0..80 = 850..930 MHz
    print(f"board: {info.get('board')}.  Dongle channel {a.ch} is taken as {freq:.3f} MHz.")
    print("step 0: configure the dongle")
    at_session(ser, setup_commands(a))
    base = f"freq={freq:.3f} bw={a.bw} sf={a.sf} cr={a.cr} pwr=4 len=20 crc=1"
    captured = None
    text = b"HUSHNET-PROBE-0123456789"

    print("step 1: dongle transmits, board listens (sync word and IQ variants)")
    syncs = list(range(256)) if a.scan_sync else [0x12, 0x34]
    hits = []
    for sync in syncs:
        for iq in ((0,) if a.scan_sync else (0, 1)):
            board_cfg(board, f"CFG {base} sync=0x{sync:02X} iq={iq}")
            board.command("MODE RX", ("MODE",), 3.0)
            board.drain()
            for _ in range(1 if a.scan_sync else 3):
                ser.write(text)
                ser.flush()
                time.sleep(0.5 if a.scan_sync else 1.2)
            rows = collect_rx(board, 0.3 if a.scan_sync else 1.5)
            if rows or not a.scan_sync:
                print(f"  sync 0x{sync:02X} iq {iq}: {len(rows)} frame(s) received by the board")
            for r in rows:
                print(f"     type={r.get('type')} len={r.get('len')} rssi={r.get('rssi')} snr={r.get('snr')} hex={r.get('hex', '')}")
                if r.get("hex") and captured is None:
                    captured = (sync, iq, r["hex"])
            if rows:
                hits.append((sync, iq, len(rows)))

    if not hits:
        print("\nRESULT: the board decoded nothing from the dongle with these settings.")
        print("Try again with --scan-sync (tries all 256 one byte sync words, about 5 minutes).")
        print("If that also finds nothing, the dongle cannot be used as a receiver for the HushNet packets.")
        board.close()
        ser.close()
        return

    sync, iq, frame_hex = captured if captured else (hits[0][0], hits[0][1], None)
    board_cfg(board, f"CFG {base} sync=0x{sync:02X} iq={iq}")
    print(f"\nstep 2: board transmits HushNet test packets with sync 0x{sync:02X} iq {iq}, dongle listens")
    ser.reset_input_buffer()
    board.command("TX 3 800", ("TX_END", "ERR", "TX_BLOCKED"), 20.0)
    got_plain = read_quiet(ser, first=1.5)
    print(f"  dongle printed {len(got_plain)} bytes" + (f": {hexdump(got_plain)}" if got_plain else ""))

    got_replay = b""
    if frame_hex:
        print("step 3: board replays the frame it captured from the dongle, dongle listens")
        ser.reset_input_buffer()
        for _ in range(3):
            board.command(f"RAW {frame_hex}", ("TX", "ERR", "TX_BLOCKED"), 10.0)
            time.sleep(0.8)
        got_replay = read_quiet(ser, first=1.5)
        print(f"  dongle printed {len(got_replay)} bytes" + (f": {hexdump(got_replay)}" if got_replay else ""))

    print("\nRESULT")
    print(f"  dongle -> board: yes, with sync word 0x{sync:02X}, IQ {'inverted' if iq else 'normal'}.")
    if frame_hex:
        on_air = bytes.fromhex(frame_hex)
        where = on_air.find(text)
        if where >= 0:
            print(f"  frame on air: {where} byte(s) of dongle header, then the text, then {len(on_air) - where - len(text)} byte(s).")
        else:
            print("  frame on air does not contain the text in clear: the dongle scrambles or encrypts the payload.")
    if got_plain:
        print("  board -> dongle: the dongle printed data for plain HushNet packets. It can be used as a receiver.")
    elif got_replay:
        print("  board -> dongle: only when the board imitates the dongle's own frame (replay worked).")
        print("  To use the dongle as a receiver the HushNet payload has to be wrapped in that frame format.")
    else:
        print("  board -> dongle: nothing printed, even for a replayed frame. Treat the dongle as transmit only for this project.")
    board.command("MODE IDLE", ("MODE",), 3.0)
    board.close()
    ser.close()


def main():
    ap = argparse.ArgumentParser(description="Waveshare USB-TO-LoRa-HF (B) helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("port", help="serial port of the dongle, for example COM7")
        p.add_argument("--sf", type=int, default=7)
        p.add_argument("--bw", type=int, default=125, help="kHz: 125, 250 or 500")
        p.add_argument("--cr", type=int, default=5, help="5..8 for 4/5..4/8")
        p.add_argument("--ch", type=int, default=18, help="channel 0..80 (18 = 868 MHz)")
        p.add_argument("--pwr", type=int, default=10, help="dBm, 10..22 (keep 10: the EU limit is 14 dBm ERP)")

    p = sub.add_parser("setup", help="set low power and the LoRa parameters")
    common(p)
    p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("at", help="send AT commands")
    p.add_argument("port")
    p.add_argument("commands", nargs="+")
    p.set_defaults(fn=cmd_at)
    p = sub.add_parser("listen", help="print received data")
    p.add_argument("port")
    p.set_defaults(fn=cmd_listen)
    p = sub.add_parser("send", help="transmit text")
    p.add_argument("port")
    p.add_argument("text")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--gap", type=float, default=1.5)
    p.set_defaults(fn=cmd_send)
    p = sub.add_parser("probe", help="interoperability experiment with a HushNet board")
    common(p)
    p.add_argument("--board", required=True, help="serial port of a board running hushnet_poc")
    p.add_argument("--scan-sync", action="store_true", help="try all 256 one byte sync words")
    p.set_defaults(fn=cmd_probe)
    a = ap.parse_args()
    if getattr(a, "pwr", 10) > 14:
        print("warning: more than 14 dBm is above the EU868 limit for most sub-bands")
    a.fn(a)


if __name__ == "__main__":
    main()
