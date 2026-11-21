"""LoRa time on air from the Semtech formula (AN1200.13 "LoRa Modem Designer's Guide",
same expression as in the SX1276 and SX1262 data sheets for SF7 to SF12).

    T_sym      = 2^SF / BW
    T_preamble = (N_preamble + 4.25) * T_sym
    N_payload  = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*IH) / (4*(SF - 2*DE))) * (CR + 4), 0)
    T_packet   = T_preamble + N_payload * T_sym

PL payload bytes, CRC 1 if the payload CRC is on, IH 1 for implicit header (0 = explicit),
DE 1 when low data rate optimisation is on, CR 1..4 for coding rate 4/5..4/8.
Low data rate optimisation is switched on when a symbol lasts 16 ms or longer
(SF11 and SF12 at 125 kHz), which is also what RadioLib does.

Command line:  python toa.py --len 20            (table for SF7..SF12)
               python toa.py --sf 9 --bw 125 --cr 5 --len 6
"""
import argparse
import math


def symbol_time_s(sf, bw_hz):
    return (2 ** sf) / float(bw_hz)


def time_on_air_s(sf, bw_hz, cr, payload_len, preamble=8, crc=True, implicit_header=False, ldro=None):
    """cr is 5..8 (coding rate 4/5..4/8). Returns seconds."""
    t_sym = symbol_time_s(sf, bw_hz)
    if ldro is None:
        ldro = t_sym >= 0.016
    de = 1 if ldro else 0
    num = 8 * payload_len - 4 * sf + 28 + (16 if crc else 0) - (20 if implicit_header else 0)
    n_payload = 8 + max(math.ceil(num / (4.0 * (sf - 2 * de))) * cr, 0)
    return (preamble + 4.25) * t_sym + n_payload * t_sym


def bitrate_bps(sf, bw_hz, cr):
    """Raw LoRa bit rate (data sheet): SF * (BW / 2^SF) * 4 / CR."""
    return sf * (bw_hz / (2 ** sf)) * 4.0 / cr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LoRa time on air (Semtech formula)")
    ap.add_argument("--sf", type=int, help="spreading factor 7..12 (default: all)")
    ap.add_argument("--bw", type=float, default=125.0, help="bandwidth in kHz (default 125)")
    ap.add_argument("--cr", type=int, default=5, help="coding rate denominator 5..8 (default 5 = 4/5)")
    ap.add_argument("--len", type=int, default=20, help="payload bytes (default 20)")
    ap.add_argument("--preamble", type=int, default=8)
    ap.add_argument("--no-crc", action="store_true")
    a = ap.parse_args()
    bw = a.bw * 1000.0
    print(f"payload {a.len} bytes, BW {a.bw:g} kHz, CR 4/{a.cr}, preamble {a.preamble}, CRC {'off' if a.no_crc else 'on'}")
    print(f"{'SF':>3} {'T_sym ms':>9} {'LDRO':>5} {'ToA ms':>9} {'bit/s':>8} {'1% duty: packets/h':>19} {'10%: packets/h':>15}")
    for sf in ([a.sf] if a.sf else range(7, 13)):
        t = time_on_air_s(sf, bw, a.cr, a.len, a.preamble, not a.no_crc)
        ts = symbol_time_s(sf, bw)
        print(f"{sf:>3} {ts * 1000:9.3f} {'on' if ts >= 0.016 else 'off':>5} {t * 1000:9.2f} {bitrate_bps(sf, bw, a.cr):8.0f} "
              f"{int(36.0 / t):>19d} {int(360.0 / t):>15d}")
