# Field notes: reproducing this on a 100G switched fabric

Second site, different fabric. Four ASUS Ascent GX10 through an Arista 7060CX-32S at
100G with passive DAC, rather than the direct-cabled 200G setup this recipe was written
on. The recipe held. What follows is what we hit that is not already in the README, and
one launcher change we think belongs upstream.

## Decode is not fabric-bound on this model

The most useful thing we can report is a negative result. At 100G, single rail, we get
numbers that match the reference build on 200G:

| Content regime | TTFT | Decode |
|---|---|---|
| structured | 0.254 s | 76.0 tok/s |
| code | 0.187 s | 71.3 tok/s |
| prose | 0.185 s | 31.6 tok/s |

Concurrency 1 / 2 / 8: 65.4, 104.2, 99.9 tok/s aggregate. KV pool 3,895,606 tokens.
DFlash2 accepts 5.43 of 7 draft tokens.

Halving fabric bandwidth changed nothing measurable in decode, which means anyone
planning a build should spend on nodes before spending on switching. Prefill of long
contexts and heavy multi-tenant load are where more fabric would still pay.

For reference, the same cluster before the RedHatAI checkpoint and DFlash2, running
ModelOpt weights with MTP-4: 64.0 structured, 46.9 code, 33.4 prose. Changing the
checkpoint and drafter moved code generation by half and left prose where it was.

## `NCCL_IB_GID_INDEX` should be looked up, not pinned

The RoCEv2/IPv4 GID does not sit at the same index on every node. On our four, rail A
was 3, 3, 3, 4. An index becomes a hole when an address is removed and re-added, so the
new GID lands in the next free slot instead of the old one.

Getting it wrong fails in two different ways, and the second one is expensive:

- a wrong explicit index gives `ibv_modify_qp failed ... local GID ::`, which is clear
- leaving it unset so NCCL auto-selects makes the collective **hang**: twenty minutes of
  no log output, GPUs at 96% utilisation

That second signature is worth internalising. **96% utilisation at 22 W is not compute.**
A GB10 loading weights draws far more. High utilisation with low power means a spin-wait
on a collective that will never complete, and `nvidia-smi --query-gpu=utilization.gpu,power.draw`
tells you in five seconds what the log will not tell you at all.

Pinning per rank is not enough either. We rebooted the fleet a few times for unrelated
reasons and one node's GID moved from 4 back to 3, so a pin that had been correct became
a pin to an empty slot. That failure looks different again: `NCCL error: unhandled system
error` at init, on one rank only, with the head reporting nothing more useful than
"WorkerProc initialization failed". Relaunching does not help, because nothing about a
relaunch changes the pin.

The launcher in this fork reads the index from sysfs before starting the container and
keeps the pinned value as a fallback. Six lines, and it prints what it chose:

```
using NCCL_IB_GID_INDEX=3
```

## Moving the weights and the image without wasting the fabric

`docker save | ssh` and rsync over SSH both encrypt, and AES on the GB10's ARM cores
tops out near 1 Gbps. On a 100G fabric that is one percent of the wire, and the rails are
an isolated L2 segment with nothing to protect the traffic from.

An rsync daemon, read-only and restricted to the rail subnets, moved a 4 GiB shard at
1977 MB/s against roughly 1 Gbps over SSH:

```ini
# /etc/rsyncd.conf
uid = mtxc
gid = mtxc
use chroot = no
read only = yes
hosts allow = 10.77.1.0/24 10.77.2.0/24
hosts deny = *

[models]
path = /var/tmp
```

The image goes the same way: `docker save` to a tar under the module root, each node
pulls it and runs `docker load`. Costs 31 GB of scratch and one extra write-read cycle,
still far quicker than encrypting the same bytes. This matters more as model sizes grow;
185 GB over SSH is most of an afternoon.

## The watchdog needs turning off before maintenance

`fleet_watchdog.sh` does exactly what it should, which is the problem. Halfway through
swapping the image and the checkpoint it saw `/health` fail, concluded the fleet was
down, tore down the containers we had just started by hand, and relaunched from its own
configuration. That configuration still named the previous launcher, which a pull from
here had deleted on the head node. Three workers came up on the old stack with no head at
all, which is worse than either state we were moving between.

Stop it first, start it after a verified launch. And after every pull, re-check the four
things the watchdog holds its own copy of: the launcher path, the node map, the SSH key,
and the health URL. A renamed launcher upstream is enough to break it silently.

## Hardware: do not use AOC in the QSFP cages

This cost us a day and is not a software problem.

The GX10 draws air in through the bottom and exhausts out the back, which is where the
QSFP cages are, so the transceivers sit in the node's own hot exhaust. With a 3.5 W AOC
in both cages the modules idle at 68 to 72 C against a 70 C warning and 75 C alarm. Under
load they cross it and the driver protects the optics:

```
mlx5_core: Port module event[error]: module 0, Cable error, High Temperature
mlx5_core enp1s0f0np0: Link down
```

A worker's rail disappearing mid-inference kills the engine with
`TimeoutError: RPC call to sample_tokens timed out`, which reads like a software fault and
is not one.

Nothing helps from software. Bringing the idle cage's interface down bought 1 C, because
the module stays powered, and `ethtool --set-module power-mode-policy` is not supported on
this device. Passive DAC dissipates 1.5 W, has no laser and no temperature sensor at all,
and the failure mode disappears with it. After the swap: 98.01 Gbps per link, 2.58 us
latency, no flaps.

Placement matters too. Ours were stacked, and the two in the middle of the row ran 81 to
84 C on the board against 70 to 73 at the edges. Bottom intake means each unit wants
clearance underneath.

## Measuring decode speed

Our first benchmark reported a flat 13.7 tok/s across every content type while the
engine's own metrics said 25 to 50. The benchmark counted SSE chunks, and with
speculative decoding one chunk carries several tokens. Ask the server instead:

```json
{"stream": true, "stream_options": {"include_usage": true}}
```

then divide `completion_tokens` by the time from first token to last.
