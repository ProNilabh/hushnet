"""Two simulated HushNet boards on pseudo terminals, to try the PC scripts without hardware.
Linux or WSL only (uses pty). It imitates the serial protocol of the hushnet_poc sketch and a
simple radio link (path loss + noise), so phy_sweep.py and benchmark.py can be exercised:

    python sim_boards.py --loss 128 &            prints three device names: board 1, board 2, dongle
    python phy_sweep.py --tx /dev/pts/3 --rx /dev/pts/4 --n 20
    python waveshare_at.py probe /dev/pts/5 --board /dev/pts/3

The numbers it produces are synthetic. It is a test tool for the scripts, not a LoRa model.
"""
import argparse
import math
import os
import pty
import random
import select
import sys
import threading
import time
import tty

from toa import time_on_air_s

SNR_LIMIT = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}   # SX1262 data sheet demodulator limits


class SimBoard:
    def __init__(self, name, tag, node, pwr_min, air):
        self.name, self.tag, self.node, self.pwr_min, self.air = name, tag, node, pwr_min, air
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.path = os.ttyname(slave)
        self.cfg = dict(freq=868100000, bw=125000, sf=7, cr=5, sync=0x12, pre=8, pwr=max(4, pwr_min), crc=1, iq=0, lna=0, len=20)
        self.mode = "IDLE"
        self.burst = 0
        self.stat = dict(rx_ok=0, rx_crc=0, lost=0, tx=0)
        self.last_seq = None
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def out(self, line):
        with self.lock:
            os.write(self.master, (line + "\r\n").encode())

    def toa(self, n):
        return time_on_air_s(self.cfg["sf"], self.cfg["bw"], self.cfg["cr"], n, self.cfg["pre"], bool(self.cfg["crc"]))

    def print_cfg(self):
        c = self.cfg
        duty = 10.0 if 869400000 <= c["freq"] <= 869650000 else 1.0
        self.out(f"CFG freq={c['freq']} bw={c['bw']} sf={c['sf']} cr={c['cr']} sync=0x{c['sync']:02X} pre={c['pre']} pwr={c['pwr']} "
                 f"chip_pwr={c['pwr'] - (13 if self.pwr_min > 0 else 0)} crc={c['crc']} iq={c['iq']} lna={c['lna']} len={c['len']} "
                 f"toa_us={round(self.toa(c['len']) * 1e6)} toa_cls_us={round(self.toa(6) * 1e6)} duty_pct={duty:.1f} band=\"simulated\"")

    def handle(self, line):
        parts = line.split()
        if not parts:
            return
        cmd, args = parts[0].upper(), parts[1:]
        if cmd == "INFO":
            self.out(f"INFO board=\"Simulated {self.tag}\" tag={self.tag} fw=sim node={self.node} chip=\"SIM\" rev=0 cores=1 cpu_mhz=160 "
                     f"flash_mb=16 psram_kb=0 sketch_kb=800 heap_free=200000 heap_min=190000")
            self.out("INFO radio=ok radiolib=0.0.0 fem=none tcxo_v=1.8 pwr_min=%d pwr_max=14" % self.pwr_min)
            self.out("INFO ml=ok model_bytes=14864 arena_used=20000 arena_size=49152 classes=4 replay_clips=8 input=\"1x49x40x1 int8\"")
            self.print_cfg()
        elif cmd == "CFG":
            for a in args:
                k, _, v = a.partition("=")
                k = k.lower()
                if k == "freq":
                    self.cfg["freq"] = round(float(v) * 1e6) if float(v) < 100000 else round(float(v))
                elif k == "bw":
                    self.cfg["bw"] = round(float(v) * 1000) if float(v) < 1000 else round(float(v))
                elif k in self.cfg:
                    self.cfg[k] = int(v, 0)
                else:
                    self.out(f"ERR unknown setting \"{k}\"")
                    return
            self.cfg["pwr"] = min(14, max(self.pwr_min, self.cfg["pwr"]))
            self.print_cfg()
        elif cmd == "MODE":
            if args:
                self.mode = args[0].upper()
            self.out(f"MODE mode={self.mode}")
        elif cmd == "STAT":
            if args and args[0].upper() == "RESET":
                self.stat.update(rx_ok=0, rx_crc=0, lost=0)
                self.last_seq = None
            s = self.stat
            self.out(f"STAT mode={self.mode} rx_ok={s['rx_ok']} rx_crc={s['rx_crc']} rx_other=0 lost={s['lost']} tx={s['tx']} "
                     f"airtime_ms_last_hour=0 budget_ms=36000 t_ms={int(time.time() * 1000) % 100000000}")
        elif cmd == "TX":
            n = int(args[0]) if args else 10
            gap = int(args[1]) / 1000.0 if len(args) > 1 else 0.1
            self.burst += 1
            self.out(f"TX_BEGIN burst={self.burst} n={n} len={self.cfg['len']} gap_ms={int(gap * 1000)}")
            for i in range(n):
                t = self.toa(self.cfg["len"])
                time.sleep(t / self.air.speed)
                meas = t * 1e6 + random.gauss(350, 40)
                self.stat["tx"] += 1
                self.out(f"TX burst={self.burst} seq={i} len={self.cfg['len']} toa_meas_us={round(meas)} toa_calc_us={round(t * 1e6)} "
                         f"total_us={round(meas + 900)} t_ms=0")
                self.air.deliver(self, i)
                time.sleep(gap / self.air.speed)
            self.out(f"TX_END burst={self.burst} sent={n} n={n}")
        elif cmd == "RAW":
            data = bytes.fromhex(args[0]) if args else b""
            self.out(f"TX type=raw len={len(data)} code=0 toa_meas_us={round(self.toa(len(data)) * 1e6)} "
                     f"toa_calc_us={round(self.toa(len(data)) * 1e6)} t_ms=0")
            self.air.raw_from_board(self, data)
        elif cmd == "INFER":
            feat, inf = (25.0, 110.0) if self.tag == "S3" else (610.0, 230.0)
            names = ["background", "chainsaw", "fire", "birds"]
            for c in range(8):
                self.out(f"INFER clip={c} src=sim.wav@2.0s true={names[c // 2]} pred={names[c // 2]} conf=0.950 correct=1 "
                         f"pc_pred={names[c // 2]} same_as_pc=1 feat_ms={feat:.1f} infer_ms={inf:.1f} feat_ndiff=0 feat_maxdiff=0 "
                         f"score_maxdiff=0 heap_free=200000")
            self.out(f"INFER_SUMMARY board={self.tag} clips=8 correct=8 same_as_pc=8 feat_ms_avg={feat:.1f} infer_ms_avg={inf:.1f} "
                     f"total_ms_avg={feat + inf:.1f} arena_used=20000 model_bytes=14864 heap_free=200000 heap_min=190000")
        else:
            self.out(f"ERR unknown command \"{cmd}\" (send HELP)")

    def receive(self, sender, seq):
        c, s = self.cfg, sender.cfg
        if self.mode != "RX" or any(c[k] != s[k] for k in ("freq", "bw", "sf", "cr", "sync", "iq")):
            return
        noise = -174 + 10 * math.log10(c["bw"]) + 6
        rssi = s["pwr"] - self.air.loss + random.gauss(0, 1.5)
        snr = min(rssi - noise, 12.0) + random.gauss(0, 0.6)
        if snr < SNR_LIMIT[c["sf"]] + random.gauss(0, 1.0):
            return
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.stat["lost"] += seq - self.last_seq - 1
        self.last_seq = seq
        self.stat["rx_ok"] += 1
        shown = max(rssi, noise + snr)
        self.out(f"RX type=test node={sender.node} burst={sender.burst} seq={seq} len={s['len']} rssi={shown:.1f} snr={snr:.2f} "
                 f"ferr_hz={round(random.gauss(-900, 60))} lost={self.stat['lost']} t_ms=0")

    def receive_foreign(self, frame, freq, sync):
        c = self.cfg
        if self.mode != "RX" or c["freq"] != freq or c["sync"] != sync or c["iq"] != 0:
            return
        self.out(f"RX type=other len={len(frame)} rssi=-40.0 snr=10.00 ferr_hz=-500 t_ms=0 hex={frame.hex().upper()}")

    def run(self):
        buf = b""
        while True:
            r, _, _ = select.select([self.master], [], [], 0.2)
            if not r:
                continue
            try:
                buf += os.read(self.master, 256)
            except OSError:
                time.sleep(0.1)
                continue
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                try:
                    self.handle(raw.decode("utf-8", "replace").strip())
                except Exception as e:      # a test tool: report and keep going
                    self.out(f"ERR simulator: {e}")


class SimDongle:
    """Imitates the serial side of the Waveshare dongle: AT mode with OK replies, and in stream mode it
    wraps the text in an invented 3 byte header. Invented behaviour, only to exercise waveshare_at.py."""
    HEADER = bytes([0x00, 0x00, 0x12])

    def __init__(self, air):
        self.air = air
        self.master, slave = pty.openpty()
        tty.setraw(slave)
        self.path = os.ttyname(slave)
        self.at_mode = False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            r, _, _ = select.select([self.master], [], [], 0.2)
            if not r:
                continue
            try:
                data = os.read(self.master, 512)
            except OSError:
                time.sleep(0.1)
                continue
            text = data.decode("latin-1")
            if text.startswith("+++"):
                self.at_mode = True
                os.write(self.master, b"OK\r\n")
            elif self.at_mode and text.upper().startswith("AT"):
                if text.upper().startswith("AT+EXIT"):
                    self.at_mode = False
                os.write(self.master, b"OK\r\n")
            elif not self.at_mode:
                for b in self.air.boards:
                    b.receive_foreign(self.HEADER + data, 868000000, 0x12)

    def from_air(self, frame):
        if frame.startswith(self.HEADER):
            os.write(self.master, frame[len(self.HEADER):])


class Air:
    def __init__(self, loss, speed):
        self.loss, self.speed, self.boards, self.dongle = loss, speed, [], None

    def raw_from_board(self, sender, data):
        if self.dongle and sender.cfg["freq"] == 868000000 and sender.cfg["sync"] == 0x12:
            self.dongle.from_air(data)

    def deliver(self, sender, seq):
        for b in self.boards:
            if b is not sender:
                b.receive(sender, seq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="two simulated HushNet boards (Linux/WSL)")
    ap.add_argument("--loss", type=float, default=100.0, help="path loss between the boards in dB")
    ap.add_argument("--speed", type=float, default=1.0, help="time acceleration factor for airtime and gaps")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    air = Air(a.loss, a.speed)
    air.boards = [SimBoard("heltec", "S3", 1, 4, air), SimBoard("nesso", "C6", 2, -9, air)]
    air.dongle = SimDongle(air)
    print(air.boards[0].path, air.boards[1].path, air.dongle.path, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)
