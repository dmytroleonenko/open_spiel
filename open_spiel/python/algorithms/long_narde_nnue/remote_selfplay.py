"""Remote LNUE shard generation for Long Narde."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def _split_hosts(hosts: str) -> List[str]:
    return [host.strip() for host in hosts.split(",") if host.strip()]


def _ssh_cmd(host: str, remote_cmd: str, ssh_args: str) -> List[str]:
    cmd = ["ssh"]
    if ssh_args:
        cmd.extend(ssh_args.split())
    cmd.append(host)
    cmd.append(remote_cmd)
    return cmd


def _copy_cmd(src: str, dst: str, rsync_args: str) -> List[str]:
    if shutil.which("rsync"):
        cmd = ["rsync", "-a"]
        if rsync_args:
            cmd.extend(rsync_args.split())
        cmd.extend([src, dst])
        return cmd
    return ["scp", src, dst]


def _assign_shards(hosts: List[str], num_shards: int) -> Dict[str, List[int]]:
    assignments: Dict[str, List[int]] = {host: [] for host in hosts}
    for shard_id in range(num_shards):
        host = hosts[shard_id % len(hosts)]
        assignments[host].append(shard_id)
    return assignments


def main() -> None:
    """Runs remote self-play shard generation."""
    # pylint: disable=too-many-locals,too-many-statements
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", required=True, help="Comma-separated host list")
    parser.add_argument("--remote_root", required=True, help="Remote repo root")
    parser.add_argument("--local_out", required=True, help="Local output directory")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--games_per_shard", type=int, required=True)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature_end", type=float, default=-1.0)
    parser.add_argument("--temperature_decay_plies", type=int, default=0)
    parser.add_argument("--root_full_depth_top_k", type=int, default=0)
    parser.add_argument("--root_reduced_depth", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--nnue", default="")
    parser.add_argument("--remote_out", default="/tmp/long_narde_shards")
    parser.add_argument("--ssh_args", default="")
    parser.add_argument("--rsync_args", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    hosts = _split_hosts(args.hosts)
    if not hosts:
        raise ValueError("No hosts specified.")

    local_out = Path(args.local_out)
    local_out.mkdir(parents=True, exist_ok=True)

    assignments = _assign_shards(hosts, args.num_shards)
    remote_bin = f"{args.remote_root}/build/games/long_narde_selfplay"

    for host, shards in assignments.items():
        if not shards:
            continue
        mkdir_cmd = f"mkdir -p {args.remote_out}"
        cmd = _ssh_cmd(host, mkdir_cmd, args.ssh_args)
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

        for shard_id in shards:
            shard_seed = args.seed + shard_id * 97
            remote_path = f"{args.remote_out}/shard_{shard_id:04d}.lnue"
            cmd_parts = [
                remote_bin,
                "--out",
                remote_path,
                "--games",
                str(args.games_per_shard),
                "--depth",
                str(args.depth),
                "--seed",
                str(shard_seed),
                "--shard_id",
                str(shard_id),
                "--workers",
                str(args.workers),
                "--chunk",
                str(args.chunk),
                "--temperature",
                str(args.temperature),
                "--temperature_end",
                str(args.temperature_end),
                "--temperature_decay_plies",
                str(args.temperature_decay_plies),
                "--root_full_depth_top_k",
                str(args.root_full_depth_top_k),
                "--root_reduced_depth",
                str(args.root_reduced_depth),
                "--alpha",
                str(args.alpha),
                "--progress",
                "0",
            ]
            if args.nnue:
                cmd_parts.extend(["--nnue", args.nnue])
            remote_cmd = " ".join(cmd_parts)
            print(f"[{host}] shard {shard_id} start", flush=True)
            cmd = _ssh_cmd(host, remote_cmd, args.ssh_args)
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
            else:
                subprocess.run(cmd, check=True)

            src = f"{host}:{remote_path}"
            dst = str(local_out / f"shard_{shard_id:04d}.lnue")
            copy_cmd = _copy_cmd(src, dst, args.rsync_args)
            if args.dry_run:
                print("DRY RUN:", " ".join(copy_cmd))
            else:
                subprocess.run(copy_cmd, check=True)
            print(f"[{host}] shard {shard_id} complete", flush=True)


if __name__ == "__main__":
    main()
