# MuZero JAX Distributed Architecture

This document explains how we scale the MuZero JAX stack from the current single-process orchestration to a fully distributed system. It captures the control/data planes, service responsibilities, deployment topologies, and the incremental milestones already captured in `TODO.md`.

> **Scope:** Actors generate trajectories via self-play, inference services run the MuZero network, the learner performs gradient updates with `pjit`, and a distributed replay/parameter plane glues everything together.

---

## 1. High-Level Components

| Component | Role | Notes |
|-----------|------|-------|
| **Learner Node(s)** | Runs the MuZero optimizer loop, samples batches from the replay service, applies gradients (eventually via `pjit`), and pushes parameter snapshots. | Only process allowed to update checkpoints. |
| **Inference Server(s)** | Stateless services hosting JAX-compiled inference. Accept batched requests over RPC, optionally across multiple hosts/devices. | On TPU: one server per chip. On GPU/CPU: multiple per host is acceptable via device contexts. |
| **Actor Nodes** | Run self-play (bootstrap + MuZero actors). They stream inference requests to the servers and upload trajectories to replay. | Default path keeps actors lightweight Python processes; future work can use Ray/MPI actors. |
| **Replay Service / Flashbax Vault** | Central buffer accessible by actors (append) and learner (sample). Manages priorities and backpressure. | Implementation can be gRPC microservice, Flashbax Vault on shared storage, or Ray-based store. |
| **Parameter Publisher** | Manages live parameter snapshots. Learner->publisher updates, inference servers subscribe, actors never read checkpoints directly. | Enables hot reload so inference weight drift stays minimal. |

---

## 2. Control & Data Planes

### 2.1 Control Plane
1. **Configuration & Orchestration** – Hydra config grows new sections (`inference.*`, `replay.*`, `publisher.*`). `run_muzero_jax.py` instantiates the right clients/server stubs depending on CLI overrides.
2. **Lifecycle Management** – Each service exposes health endpoints (`/ready`, `/live`). Orchestrator (or an external supervisor) can roll out new versions and drain queues cleanly.
3. **Monitoring** – All nodes emit metrics (prometheus/json) for queue depth, request latency, gradient stats, actor throughput, etc.

### 2.2 Data Plane
1. **Inference RPC** – Actors send `InferenceRequest(observation, optional hidden state, metadata)` to inference servers. Servers batch requests (size + timeout) before invoking JAX. Responses include policy logits, value, reward, projection, hidden states.
2. **Replay Transport** – Actors push completed trajectories (`observations`, `actions`, `rewards`, `target_values`, `policy_targets`, `length`, `priorities`). Service enforces capacity, priority sampling, and optional shard replication.
3. **Parameter Streaming** – Learner publishes serialized weights after checkpoints (or at a higher cadence). Inference servers acknowledge receipt and swap params using zero-copy if possible. Actors only talk to inference servers, never to the publisher.

---

## 3. Deployment Topologies

| Topology | Description | Targets |
|----------|-------------|---------|
| **Local Dev (Single Host)** | `inference.enable_local_batching=true` spins up background batching servers inside the process. Actors remain threads/processes; no external RPC needed. | Quick validation / CI. |
| **Single Host, Multi-Process** | Separate Python processes for learner, actors, and inference servers. Communication uses gRPC or multiprocessing queues over loopback. | Pre-production rehearsal. |
| **Multi-Host Cluster** | Dedicated hosts for learners (often TPU/GPU), inference servers (GPU-heavy), and actor pools (CPU-heavy). Replay service runs on redundant nodes backed by NFS/S3. Parameter publisher sits with the learner. | Production-scale experiments. |
| **Ray/MPI (Task 30 in `TODO.md`)** | Wrap actors/inference/replay in Ray actors or MPI ranks for auto-scaling and scheduling. | Optional advanced scaling once baseline distributed stack is stable. |

---

## 4. Implementation Milestones

These correspond to `TODO.md` Phase 2 tasks:

1. **Task 11 – Replay & Parameter Plane**
   - Deploy Flashbax Vault or a gRPC replay service with `AppendTrajectory`, `SampleBatch`, `UpdatePriorities`.
   - Add parameter publisher (simple gRPC service or pub/sub). Learner pushes weights each checkpoint interval; inference servers pull/subscribe.
   - Tests: `tests/test_distributed_replay_buffer.py` to simulate concurrent writers/readers; `tests/services/test_parameter_publisher.py` once the publisher exists.
   - **Status (Nov 16, 2025):** In-memory + gRPC prototypes live in `services/replay_service.py` and `services/parameter_publisher.py` with expanded edge coverage (`tests/services/test_replay_service_extras.py`, `test_replay_service_edge_cases.py`, `test_parameter_publisher_edge_cases.py`, `test_parameter_publisher_subscribe.py`). `build_replay_buffer` routes configs with `replay_buffer.remote_enabled=true` to a `RemoteReplayBufferAdapter`; actors refresh params before episodes when publisher is enabled; learner publishes params each training step (`publisher.publish_interval`) and on shutdown. End-to-end remote wiring smoke test: `tests/test_orchestrator_remote_end_to_end.py`. **Testing note:** gRPC subscribe tests publish once before subscribing to avoid transport-level blocking seen on macOS/Metal; production push path is unchanged.

2. **Task 12 – Distributed Actors & Remote Inference**
   - Actors use the RPC client instead of local networks (today’s `LocalBatchingInferenceClient` is already a drop-in). Swap to a gRPC client when the server exists.
   - Add config entries (`inference.remote_enabled=true`, `inference.rpc_endpoint=grpc://...`) to route actors to remote servers.
   - Tests: `tests/self_play/test_distributed_actor.py` verifying actors poll parameter publisher and push trajectories via RPC.

3. **Task 13 – Distributed Learner (`pjit`)**
   - Introduce device meshes, sharded optimizer states, and distributed checkpointing.
   - Learner samples from replay RPC, publishes parameters after each optimizer step or checkpoint, and exposes health metrics.
   - Tests: `tests/training/test_distributed_trainer.py` covering `pjit` sharding and checkpoint restore semantics.

4. **Task 30 – Advanced Orchestration (Optional)**
   - Ray-based deployment (actors, inference, replay, parameter server) with auto-scaling and placement groups.
   - Benchmarks comparing Ray vs base RPC stack on a standard game (e.g., Breakthrough).

---

## 5. APIs & Message Shapes

### 5.1 Inference RPC (proto sketch)
```proto
message InferenceRequest {
  repeated float observation = 1;       // flattened obs
  repeated float hidden_state = 2;      // optional
  repeated float reward_hidden = 3;     // optional
  uint32 player_id = 4;
  uint64 request_id = 5;
}

message InferenceResponse {
  repeated float policy_logits = 1;
  float value = 2;
  float reward = 3;
  repeated float hidden_state = 4;
  repeated float reward_hidden = 5;
  uint64 request_id = 6;
}
```
Servers batch multiple `InferenceRequest`s before calling JAX; `request_id` lets clients correlate responses.

### 5.2 Replay RPC
```proto
message Trajectory {
  repeated float observations = 1;  // flattened (len = length * obs_dim)
  repeated int32 actions = 2;
  repeated float rewards = 3;
  repeated float target_values = 4;
  repeated float policy_targets = 5;
  uint32 length = 6;
  repeated float priorities = 7;
}
```

---

## 6. Testing Strategy

| Layer | Test Focus | File(s) |
|-------|------------|---------|
| Services (Inference, Replay, Publisher) | Batching correctness, timeout handling, RPC serialization/deserialization, backpressure. | `tests/services/test_inference_server.py`, `tests/services/test_replay_service.py`, etc. |
| Actors | Parameter refresh cadence, inference client injection, trajectory upload. | `tests/self_play/test_distributed_actor.py`. |
| Learner | `pjit` correctness, distributed checkpointing, sampling via RPC. | `tests/training/test_distributed_trainer.py`. |
| Integration | Mini cluster with in-process RPC stubs verifying end-to-end training progress. | `tests/test_orchestrator_async.py` (future distributed variants). |
| Soak | Multi-process replay + publisher via gRPC with basic QPS sanity checks. | `tests/services/test_replay_publisher_soak.py`. |
| Localhost preset | Orchestrator + actors using gRPC endpoints on localhost. | `tests/test_orchestrator_localhost_remote.py`, preset `configs/presets/localhost_remote.yaml`. |

CI should pin `JAX_PLATFORM_NAME=cpu` for deterministic reproducibility; GPU/TPU tests live behind opt-in markers.

---

## 7. Operational Notes

1. **Resource Allocation:** Keep TPU/Metal rules in mind. TPU enforces single process per chip; inference + learner share a process but expose RPC endpoints to actors. On GPU/CPU you can host multiple inference servers per device, but monitor memory pressure (JAX preallocate flags).
2. **Config Overrides:** Hydra overrides drive everything (`inference.remote_enabled`, `replay.rpc_endpoint`, `resource_management.concurrent`, etc.). Documented commands live in `AGENTS.md`.
3. **Logging Discipline:** Redirect long-running commands to `/tmp/muzero_bt_debug_<slug>.log` and use `wc -l`/`tail` to keep context windows manageable (per repo policy).
4. **Deployment Recipes:** 
   - Local dev: `python -q ... inference.enable_local_batching=true resource_management.concurrent=true actors.num_actors=4`.
   - Multi-host (placeholder): use a supervisor (K8s, Ray, SLURM) to launch `learner`, `inference_server`, `replay_server`, and `actor` binaries with the appropriate Hydra overrides.

---

## 8. Open Questions

1. **Transport Choice:** gRPC vs Ray vs ZeroMQ? gRPC provides language interoperability; Ray simplifies scaling. Decision pending once MVP RPC path exists.
2. **Replay Consistency:** Flashbax Vault vs custom service—Vault gives persistence out of the box but requires shared storage; custom RPC eases deployment but needs replication logic.
3. **Security/Isolation:** When running across multiple machines/clouds, we may need TLS for RPC channels and signed checkpoints.

All future design/implementation discussions around distributed training should reference this document, and updates must stay synchronized with `TODO.md` and `AGENTS.md`.
