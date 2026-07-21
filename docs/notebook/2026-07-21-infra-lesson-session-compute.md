# 2026-07-21 (evening) — Infra lesson: session-scoped compute

The remote container suspends shortly after the session goes idle and kills
orphaned (nohup'd) processes. Two lens-fit runs died this way; progress was
protected both times by jlens's per-prompt checkpointing (n_done=9 held
across two container restarts — the checkpoint design is earning its keep).

Mitigations now in place, in order of preference:

1. Long jobs run as harness-tracked background tasks (not nohup), which the
   harness keeps running across turns and reports on exit.
2. A persistent log monitor emits an event every ~5 fitted prompts; each
   event wakes the session, which keeps the container from idling long.
3. Server-side scheduled triggers (~45 min) as the outer safety net: verify
   liveness (process AND progress AND container uptime — a live-looking
   process with frozen progress means a restart happened), restart from
   checkpoint if needed.
4. Fallback if all else fails: "babysit" mode — keep turns alive with
   bounded foreground wait-loops so the fit only computes during awake time.

General rule for this environment: any computation longer than ~10 minutes
must be checkpointed and resumable, because the compute substrate is only
reliably alive while the session is.
