"""LoRa physical layer sweep with two HushNet boards on one PC.

One board transmits bursts of numbered test packets, the other receives them. For every
spreading factor and transmit power the script logs RSSI, SNR, frequency error, packet error
rate and the measured time on air, and compares the airtime with the Semtech formula.

    python phy_sweep.py --tx COM5 --rx COM6
    python phy_sweep.py --tx COM5 --rx COM6 --sf 7-12 --pwr 4,9,14 --n 30 --len 20 --note "corridor 25 m"

Defaults: 869.525 MHz (the 869.4 to 869.65 MHz sub-band allows 10 % duty cycle, so the sweep
finishes in minutes), 125 kHz, coding rate 4/5, 20 byte payload, 30 packets per setting.

Output files (prefix set with --out, default "sweep"):
    sweep_packets.csv   one row per transmitted packet
    sweep_summary.csv   one row per (SF, power) setting
    sweep.png           plots (needs matplotlib; skipped if it is not installed)

Both boards must run the hushnet_poc sketch. Close the Arduino Serial Monitor first.
"""
import argparse
import csv
import re
import statistics
import sys
import time

from hn_serial import Board, parse_line
from toa import bitrate_bps, time_on_air_s


def parse_int_list(text):
    """'7-12' -> [7..12], '4,9,14' -> [4, 9, 14], '-5,0' -> [-5, 0]"""
    out = []
    for part in text.split(","):
        part = part.strip()
        m = re.match(r"^(-?\d+)-(-?\d+)$", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part:
            out.append(int(part))
    return out


def must(board, cmd, tags, timeout=5.0):
    tag, fields, seen = board.command(cmd, tuple(tags) + ("ERR",), timeout)
    if tag is None:
        sys.exit(f"{board.name}: no answer to '{cmd}'")
    if tag == "ERR":
        sys.exit(f"{board.name}: '{cmd}' -> {seen[-1][1]}")
    return fields


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="LoRa PHY sweep: SF and TX power against RSSI, SNR, PER and airtime")
    ap.add_argument("--tx", required=True, help="serial port of the transmitting board")
    ap.add_argument("--rx", required=True, help="serial port of the receiving board")
    ap.add_argument("--sf", default="7-12", help="spreading factors, for example 7-12 or 7,9,12")
    ap.add_argument("--pwr", default="4,9,14", help="antenna power values in dBm (the board clamps to what it can do)")
    ap.add_argument("--n", type=int, default=30, help="packets per setting")
    ap.add_argument("--len", type=int, default=20, help="payload bytes")
    ap.add_argument("--freq", type=float, default=869.525, help="MHz")
    ap.add_argument("--bw", type=float, default=125.0, help="kHz")
    ap.add_argument("--cr", type=int, default=5, help="5..8 for coding rate 4/5..4/8")
    ap.add_argument("--gap", type=int, default=100, help="pause between packets in ms")
    ap.add_argument("--lna", type=int, default=0, help="Heltec V4.3 receiver: 1 = use the front end LNA")
    ap.add_argument("--out", default="sweep", help="prefix of the output files")
    ap.add_argument("--note", default="", help="free text stored in every CSV row (distance, place, antennas)")
    a = ap.parse_args()

    sfs = parse_int_list(a.sf)
    pwrs = parse_int_list(a.pwr)
    bw_hz = a.bw * 1000.0
    run_id = time.strftime("%Y%m%d_%H%M%S")
    raw = open(f"{a.out}_raw.log", "a")
    tx = Board(a.tx, "TX", raw_log=raw)
    rx = Board(a.rx, "RX", raw_log=raw)
    info_tx, info_rx = tx.hello(), rx.hello()
    print(f"TX board: {info_tx.get('board')} (node {info_tx.get('node')})   RX board: {info_rx.get('board')} (node {info_rx.get('node')})")
    if a.tx == a.rx:
        sys.exit("--tx and --rx must be two different ports")

    airtime = sum(time_on_air_s(sf, bw_hz, a.cr, a.len) for sf in sfs) * a.n * len(pwrs)
    print(f"planned transmit time: {airtime:.0f} s in total "
          f"(limit per hour: 36 s in a 1 % sub-band, 360 s in the 10 % sub-band; the firmware enforces it)")

    must(tx, "MODE IDLE", ["MODE"])
    base = f"freq={a.freq:.6f} bw={a.bw:g} cr={a.cr} len={a.len}"
    packets, summary = [], []
    stop = False
    try:
        for pwr in pwrs:
            for sf in sfs:
                must(rx, f"CFG {base} sf={sf} lna={a.lna}", ["CFG"])
                must(rx, "MODE RX", ["MODE"])
                must(rx, "STAT RESET", ["STAT"])
                cfg = must(tx, f"CFG {base} sf={sf} pwr={pwr}", ["CFG"])
                pwr_actual = int(cfg.get("pwr", pwr))
                toa_formula = time_on_air_s(sf, bw_hz, a.cr, a.len)
                tx.drain()
                rx.drain()
                tx.send(f"TX {a.n} {a.gap}")
                sent, got, burst, crc_err, blocked = {}, {}, None, 0, False
                deadline = time.time() + a.n * (toa_formula + a.gap / 1000.0 + 0.4) + 15.0
                end_seen = None
                while time.time() < deadline:
                    for b in (tx, rx):
                        item = b.read(0.02)
                        if item is None:
                            continue
                        tag, f = parse_line(item[1])
                        if b is tx:
                            if tag == "TX_BEGIN":
                                burst = f.get("burst")
                            elif tag == "TX" and "seq" in f:
                                sent[int(f["seq"])] = f
                            elif tag == "TX_BLOCKED":
                                blocked = True
                                print(f"  transmit blocked by the board: {item[1]}")
                            elif tag in ("TX_ERR", "ERR"):
                                print(f"  {item[1]}")
                            elif tag == "TX_END":
                                end_seen = time.time()
                        else:
                            if tag == "RX" and f.get("type") == "test" and (burst is None or f.get("burst") == burst):
                                got[int(f["seq"])] = f
                            elif tag == "RX" and f.get("type") == "crc_error":
                                crc_err += 1
                    if end_seen and time.time() - end_seen > toa_formula + 0.6:
                        break
                rssi = [float(got[s]["rssi"]) for s in got if s in sent]
                snr = [float(got[s]["snr"]) for s in got if s in sent]
                ferr = [float(got[s]["ferr_hz"]) for s in got if s in sent]
                meas = [int(sent[s]["toa_meas_us"]) / 1000.0 for s in sent]
                calc = [int(sent[s]["toa_calc_us"]) / 1000.0 for s in sent]
                n_sent, n_ok = len(sent), len(rssi)
                per = 100.0 * (1.0 - n_ok / n_sent) if n_sent else float("nan")
                row = dict(run_id=run_id, note=a.note, tx_board=info_tx.get("tag"), rx_board=info_rx.get("tag"),
                           freq_mhz=a.freq, bw_khz=a.bw, cr=a.cr, sf=sf, len=a.len, pwr_req_dbm=pwr, pwr_dbm=pwr_actual,
                           sent=n_sent, received=n_ok, crc_errors=crc_err, per_pct=round(per, 2),
                           rssi_mean=round(mean(rssi), 2), rssi_min=min(rssi) if rssi else "", rssi_max=max(rssi) if rssi else "",
                           snr_mean=round(mean(snr), 2), snr_min=min(snr) if snr else "",
                           ferr_mean_hz=round(mean(ferr), 0) if ferr else "",
                           toa_meas_ms=round(mean(meas), 3), toa_meas_std_ms=round(statistics.pstdev(meas), 3) if len(meas) > 1 else 0,
                           toa_radiolib_ms=round(mean(calc), 3), toa_formula_ms=round(toa_formula * 1000.0, 3),
                           meas_minus_formula_ms=round(mean(meas) - toa_formula * 1000.0, 3) if meas else "",
                           bitrate_bps=round(bitrate_bps(sf, bw_hz, a.cr), 1))
                summary.append(row)
                for s in sorted(sent):
                    g = got.get(s)
                    packets.append(dict(run_id=run_id, note=a.note, sf=sf, bw_khz=a.bw, cr=a.cr, len=a.len, pwr_dbm=pwr_actual,
                                        burst=burst, seq=s, received=1 if g else 0,
                                        rssi=g["rssi"] if g else "", snr=g["snr"] if g else "", ferr_hz=g["ferr_hz"] if g else "",
                                        toa_meas_us=sent[s]["toa_meas_us"], toa_radiolib_us=sent[s]["toa_calc_us"],
                                        toa_formula_us=round(toa_formula * 1e6)))
                print(f"SF{sf:<2d} {pwr_actual:>3d} dBm  sent {n_sent:>3d}  received {n_ok:>3d}  PER {per:5.1f} %  "
                      f"RSSI {mean(rssi):7.1f} dBm  SNR {mean(snr):6.2f} dB  ToA measured {mean(meas):8.2f} ms  "
                      f"formula {toa_formula * 1000:8.2f} ms")
                if blocked or n_sent == 0:
                    stop = True
                    break
            if stop:
                print("stopped: the transmitter refused to send (duty cycle budget or channel). See the message above.")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping the burst")
        tx.send("STAT")
    finally:
        try:
            tx.send("MODE IDLE")
        except Exception:
            pass
        time.sleep(0.2)
        tx.close()
        rx.close()
        raw.close()

    if packets:
        with open(f"{a.out}_packets.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(packets[0].keys()))
            w.writeheader()
            w.writerows(packets)
    if summary:
        with open(f"{a.out}_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"saved {a.out}_packets.csv and {a.out}_summary.csv")
        plot(summary, a.out)


def plot(summary, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed, no plot (pip install matplotlib)")
        return
    pwrs = sorted({r["pwr_dbm"] for r in summary})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    first = [r for r in summary if r["pwr_dbm"] == pwrs[0]]
    sf = [r["sf"] for r in first]
    ax[0][0].plot(sf, [r["toa_formula_ms"] for r in first], "k--", label="Semtech formula")
    ax[0][0].plot(sf, [r["toa_meas_ms"] for r in first], "o-", label="measured")
    ax[0][0].set_yscale("log")
    ax[0][0].set_title("Time on air")
    ax[0][0].set_ylabel("ms")
    ax[0][0].legend()
    for p in pwrs:
        rows = [r for r in summary if r["pwr_dbm"] == p]
        x = [r["sf"] for r in rows]
        ax[0][1].plot(x, [r["rssi_mean"] for r in rows], "o-", label=f"{p} dBm")
        ax[1][0].plot(x, [r["snr_mean"] for r in rows], "o-", label=f"{p} dBm")
        ax[1][1].plot(x, [r["per_pct"] for r in rows], "o-", label=f"{p} dBm")
    ax[0][1].set_title("RSSI (mean)")
    ax[0][1].set_ylabel("dBm")
    ax[1][0].set_title("SNR (mean)")
    ax[1][0].set_ylabel("dB")
    ax[1][1].set_title("Packet error rate")
    ax[1][1].set_ylabel("%")
    for row in ax:
        for x in row:
            x.set_xlabel("spreading factor")
            x.grid(True, alpha=0.3)
    ax[0][1].legend(title="TX power")
    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=130)
    print(f"saved {prefix}.png")


if __name__ == "__main__":
    main()
