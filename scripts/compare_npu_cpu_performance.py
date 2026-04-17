#!/usr/bin/env python3
"""Run NPU and CPU backend benchmarks and print a comparison report.

Example:
    python scripts/compare_npu_cpu_performance.py --exe build/bin/Debug/npu_inference_engine.exe --iters 200 --warmup 20

Power measurement:
    Uses the Windows Energy Metering Interface (EMI) via PDH performance counters.
    Works on both AC and battery.  Reports per-component and system-level watts.
    Units: EMI counters accumulate energy in nanojoules; power = delta_nJ / elapsed_s / 1e9.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

RESULT_RE = re.compile(r"^BENCHMARK_RESULT\s+(.*)$")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")

# ---------------------------------------------------------------------------
# Real-time power sampler via Windows EMI (Energy Metering Interface)
# ---------------------------------------------------------------------------
# PDH counters: \Energy Meter(<channel>)\Energy
# Units: cumulative nanojoules.  Power = delta_nJ / elapsed_s / 1e9
# Channels on Snapdragon X Elite (may vary by machine):
#   cpu_cluster_0, cpu_cluster_1, cpu_cluster_2, gpu, sys
# 'sys' = total SoC power draw.

_EMI_CHANNELS = ["cpu_cluster_0", "cpu_cluster_1", "cpu_cluster_2", "gpu", "sys"]

# PowerShell loop: samples every 200 ms, emits one line per tick:
#   <timestamp_100ns_ticks> <ch0_nj> <ch1_nj> ... <chN_nj>
_EMI_PS_SCRIPT = (
    "while ($true) {"
    " $ts = [datetime]::UtcNow.Ticks;"
    " $vals = @();"
    " foreach ($ch in @('cpu_cluster_0','cpu_cluster_1','cpu_cluster_2','gpu','sys')) {"
    "  try { $v = (Get-Counter \"\\Energy Meter($ch)\\Energy\" -EA Stop).CounterSamples[0].CookedValue; $vals += $v }"
    "  catch { $vals += -1 }"
    " };"
    " Write-Output (\"$ts \" + ($vals -join ' '));"
    " Start-Sleep -Milliseconds 200"
    "}"
)


class EmiPowerSampler:
    """Samples Windows EMI energy counters in a background thread.

    Computes instantaneous power for each channel as:
        W = (energy_end_nJ - energy_start_nJ) / elapsed_s / 1e9

    Works on AC and battery; requires only the built-in Windows Energy Meter
    PDH provider (present on Snapdragon X Elite and many other ARM devices).
    """

    def __init__(self) -> None:
        self._readings: List[tuple] = []  # (ticks, [nJ per channel])
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _EMI_PS_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                ticks = int(parts[0])
                vals = [float(p) for p in parts[1:]]
                with self._lock:
                    self._readings.append((ticks, vals))
            except (ValueError, IndexError):
                pass

    def _watts_per_channel(self) -> Optional[List[float]]:
        """Mean watts per channel averaged over all consecutive sample pairs."""
        with self._lock:
            readings = list(self._readings)
        if len(readings) < 2:
            return None

        n_ch = len(_EMI_CHANNELS)
        sums = [0.0] * n_ch
        count = 0
        for i in range(len(readings) - 1):
            t0, v0 = readings[i]
            t1, v1 = readings[i + 1]
            elapsed_s = (t1 - t0) / 1e7  # 100ns ticks -> seconds
            if elapsed_s <= 0:
                continue
            for c in range(n_ch):
                if len(v0) <= c or len(v1) <= c:
                    continue
                if v0[c] < 0 or v1[c] < 0:
                    continue
                sums[c] += (v1[c] - v0[c]) / elapsed_s / 1e9  # nJ/s -> W
            count += 1

        if count == 0:
            return None
        return [s / count for s in sums]

    def _idx(self, name: str) -> int:
        return _EMI_CHANNELS.index(name)

    @property
    def avg_watts_sys(self) -> Optional[float]:
        w = self._watts_per_channel()
        return w[self._idx("sys")] if w is not None else None

    @property
    def avg_watts_cpu(self) -> Optional[float]:
        """Sum of all cpu_cluster channels."""
        w = self._watts_per_channel()
        if w is None:
            return None
        return sum(w[i] for i, ch in enumerate(_EMI_CHANNELS) if ch.startswith("cpu_cluster"))

    @property
    def avg_watts_gpu(self) -> Optional[float]:
        w = self._watts_per_channel()
        return w[self._idx("gpu")] if w is not None else None

    @property
    def peak_watts_sys(self) -> Optional[float]:
        """Peak instantaneous system watts across all sample pairs."""
        with self._lock:
            readings = list(self._readings)
        if len(readings) < 2:
            return None
        idx = self._idx("sys")
        peak = 0.0
        for i in range(len(readings) - 1):
            t0, v0 = readings[i]
            t1, v1 = readings[i + 1]
            elapsed_s = (t1 - t0) / 1e7
            if elapsed_s <= 0 or len(v0) <= idx or v0[idx] < 0 or v1[idx] < 0:
                continue
            peak = max(peak, (v1[idx] - v0[idx]) / elapsed_s / 1e9)
        return peak if peak > 0 else None


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_result(stdout: str) -> Dict[str, float | str]:
    for line in stdout.splitlines():
        match = RESULT_RE.match(line.strip())
        if not match:
            continue
        payload = match.group(1)
        values: Dict[str, float | str] = {}
        for key, raw_value in KV_RE.findall(payload):
            if key == "backend":
                values[key] = raw_value
            else:
                try:
                    values[key] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"Could not parse numeric value for {key}: {raw_value}") from exc
        return values
    raise ValueError("Missing BENCHMARK_RESULT line in executable output.")


# ---------------------------------------------------------------------------
# Backend runner
# ---------------------------------------------------------------------------

def run_backend(exe: Path, backend: str, warmup: int, iters: int, repeats: int) -> Dict[str, float | str]:
    runs: List[Dict[str, float | str]] = []
    sys_samples: List[float] = []
    cpu_samples: List[float] = []
    gpu_samples: List[float] = []
    peak_samples: List[float] = []

    for idx in range(repeats):
        cmd = [str(exe), "--benchmark", "--backend", backend,
               "--warmup", str(warmup), "--iters", str(iters)]

        sampler = EmiPowerSampler()
        sampler.start()
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        sampler.stop()

        if completed.returncode != 0:
            raise RuntimeError(
                f"Backend '{backend}' run {idx + 1}/{repeats} failed "
                f"(exit {completed.returncode}).\nSTDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        runs.append(parse_result(completed.stdout))

        if sampler.avg_watts_sys is not None:
            sys_samples.append(sampler.avg_watts_sys)
        if sampler.avg_watts_cpu is not None:
            cpu_samples.append(sampler.avg_watts_cpu)
        if sampler.avg_watts_gpu is not None:
            gpu_samples.append(sampler.avg_watts_gpu)
        if sampler.peak_watts_sys is not None:
            peak_samples.append(sampler.peak_watts_sys)

    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, float)]
    merged: Dict[str, float | str] = {"backend": backend}
    for key in numeric_keys:
        merged[key] = statistics.mean(float(run[key]) for run in runs)

    nan = float("nan")
    merged["avg_watts_sys"] = statistics.mean(sys_samples) if sys_samples else nan
    merged["avg_watts_cpu"] = statistics.mean(cpu_samples) if cpu_samples else nan
    merged["avg_watts_gpu"] = statistics.mean(gpu_samples) if gpu_samples else nan
    merged["peak_watts_sys"] = max(peak_samples) if peak_samples else nan
    merged["power_ok"] = "yes" if sys_samples else "no"

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_change(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def fw(val: float) -> str:
    return f"{val:.3f} W" if not math.isnan(val) else "N/A"


def fj(val: Optional[float]) -> str:
    return f"{val:.2f} inf/J" if val is not None else "N/A"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare QNN NPU and CPU backend performance.")
    parser.add_argument(
        "--exe", type=Path,
        default=Path("build") / "bin" / "Debug" / "npu_inference_engine.exe",
        help="Path to built npu_inference_engine executable.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2
    if args.iters <= 0:
        print("--iters must be > 0", file=sys.stderr)
        return 2
    if args.repeats <= 0:
        print("--repeats must be > 0", file=sys.stderr)
        return 2
    if not args.exe.exists():
        print(f"Executable not found: {args.exe}\nBuild first.", file=sys.stderr)
        return 2

    print(f"Running benchmark for NPU backend ({args.repeats} repeats)...")
    npu = run_backend(args.exe, "npu", args.warmup, args.iters, args.repeats)

    print(f"Running benchmark for CPU backend ({args.repeats} repeats)...")
    cpu = run_backend(args.exe, "cpu", args.warmup, args.iters, args.repeats)

    # --- latency ---
    npu_avg  = float(npu["avg_ms"])
    cpu_avg  = float(cpu["avg_ms"])
    npu_tp   = float(npu["throughput_fps"])
    cpu_tp   = float(cpu["throughput_fps"])
    npu_cold = float(npu["init_ms"]) + float(npu["build_ms"]) + float(npu["cold_ms"])
    cpu_cold = float(cpu["init_ms"]) + float(cpu["build_ms"]) + float(cpu["cold_ms"])

    # --- power ---
    npu_sys_w  = float(npu["avg_watts_sys"])
    cpu_sys_w  = float(cpu["avg_watts_sys"])
    npu_cpu_w  = float(npu["avg_watts_cpu"])
    cpu_cpu_w  = float(cpu["avg_watts_cpu"])
    npu_gpu_w  = float(npu["avg_watts_gpu"])
    cpu_gpu_w  = float(cpu["avg_watts_gpu"])
    npu_peak_w = float(npu["peak_watts_sys"])
    cpu_peak_w = float(cpu["peak_watts_sys"])
    power_ok   = npu["power_ok"] == "yes" and cpu["power_ok"] == "yes"

    # --- derived ---
    speedup  = cpu_avg / npu_avg if npu_avg > 0 else 0.0
    lat_red  = -pct_change(cpu_avg, npu_avg)
    tp_gain  = pct_change(cpu_tp, npu_tp)
    npu_eff  = npu_tp / npu_cold if npu_cold > 0 else 0.0
    cpu_eff  = cpu_tp / cpu_cold if cpu_cold > 0 else 0.0
    eff_gain = pct_change(cpu_eff, npu_eff)

    npu_ipj = npu_tp / npu_sys_w if (power_ok and not math.isnan(npu_sys_w) and npu_sys_w > 0) else None
    cpu_ipj = cpu_tp / cpu_sys_w if (power_ok and not math.isnan(cpu_sys_w) and cpu_sys_w > 0) else None

    # --- output ---
    print(f"\n=== Backend Comparison (means across {args.repeats} repeats) ===")
    print(f"Warmup: {args.warmup}  |  Timed iters: {args.iters}")
    if not power_ok:
        print("[!] EMI power counters unavailable on this machine.")

    print("\nLatency & Throughput")
    print(f"  NPU: avg={npu_avg:.4f} ms  p95={float(npu['p95_ms']):.4f} ms  "
          f"throughput={npu_tp:.2f} inf/s  cold_total={npu_cold:.4f} ms")
    print(f"  CPU: avg={cpu_avg:.4f} ms  p95={float(cpu['p95_ms']):.4f} ms  "
          f"throughput={cpu_tp:.2f} inf/s  cold_total={cpu_cold:.4f} ms")

    print("\nPower (Windows EMI — direct hardware nanojoule counters, works on AC)")
    print(f"  NPU: sys={fw(npu_sys_w)}  cpu={fw(npu_cpu_w)}  gpu={fw(npu_gpu_w)}"
          f"  sys_peak={fw(npu_peak_w)}  efficiency={fj(npu_ipj)}")
    print(f"  CPU: sys={fw(cpu_sys_w)}  cpu={fw(cpu_cpu_w)}  gpu={fw(cpu_gpu_w)}"
          f"  sys_peak={fw(cpu_peak_w)}  efficiency={fj(cpu_ipj)}")

    print("\nDifferences")
    print(f"  Latency speedup (CPU avg / NPU avg):   {speedup:.3f}x")
    print(f"  Latency reduction (NPU vs CPU):        {lat_red:.2f}%")
    print(f"  Throughput gain (NPU vs CPU):          {tp_gain:.2f}%")
    print(f"  Cold-path efficiency gain:             {eff_gain:.2f}%")
    if npu_ipj is not None and cpu_ipj is not None and cpu_ipj > 0:
        print(f"  Power efficiency gain (inf/J):         {pct_change(cpu_ipj, npu_ipj):.2f}%")

    print("\nNote: Power = per-component SoC draw from Windows EMI (nJ counters).")
    print("      'sys' covers total SoC. Values include background OS activity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
