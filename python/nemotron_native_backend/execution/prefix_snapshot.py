import hashlib
import time

WIDTHS = (1, 2, 4, 8, 16, 64)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class PrefixSnapshot:
    def __init__(self, engine):
        self.engine = engine
        self.n = n = engine.core.neural
        self.t = engine.t
        require(engine.batch in WIDTHS and n.batch == engine.batch, "Unsupported FP32 prefix width")
        require(not n.started and not n.closed, "Prefix preparation requires a fresh idle owner")
        require(
            not getattr(n, "_r8_initial_state", None),
            "Canonical-state entries need their existing cache",
        )
        require(not hasattr(engine, "_fp32_prefix_snapshot"), "Prefix owner is already installed")
        require(
            n.cache["ssm_dtype"] == "FP32" and n.cache["conv_dtype"] == "BF16",
            "Unrecognized state contract",
        )
        require(
            n.info.ssm_slot_elements == 196608 and n.info.conv_slot_elements == 10752,
            "Unexpected recurrent allocation geometry",
        )
        require(
            all(
                n.bindings.ssm_slot_stride[i] == 196608 and n.bindings.conv_slot_stride[i] == 10752
                for i in range(14)
            ),
            "Recurrent state strides differ from the source contract",
        )
        self.original = n.prefill_lib.mk2_native_prefill_run
        self.original_close = n.close
        self.identity = self.owner_identity()
        require(
            n.inputs.prompt.is_contiguous()
            and n.inputs.prompt.dtype == self.t.bfloat16
            and str(n.inputs.prompt.device) == "cuda:0"
            and tuple(n.inputs.prompt.shape) == (n.config.prompt_positions, n.batch, 1536),
            "Native prefill requires fixed contiguous BF16 position-major CUDA0 prompt",
        )
        self.prompt_allocation = self.prompt_layout()
        self.prompt_identity = self.prompt_key()
        self.closed = False
        self.views = self.state_views()
        self.snapshots = []
        started = time.monotonic()
        with self.t.cuda.stream(n.stream):
            n._check(
                self.original(n.prefill_handle, n.stream.cuda_stream),
                "FP32 prefix snapshot preparation",
                n.prefill_lib,
            )
            self.snapshots = [value.clone() for value in self.views]
        n.stream.synchronize()
        self.receipt = dict(
            schema="nemotron.r22.fp32-prefix-snapshot.v1",
            status="PREPARED_UNQUALIFIED",
            width=engine.batch,
            state="FP32-private",
            conv="BF16",
            kv="BF16-valid-prefix-only",
            neural_implementation_changed=False,
            request_text_cached=False,
            output_cached=False,
            context_noise_draw_removed=False,
            bytes=sum(value.numel() * value.element_size() for value in self.snapshots),
            preparation_seconds=time.monotonic() - started,
            prefill_sha256=n.prefill_library_sha256,
            stage0_sha256=n.library_sha256,
            prompt_sha256=hashlib.sha256(
                n.inputs.prompt.cpu().view(self.t.uint8).numpy().tobytes()
            ).hexdigest(),
            hits=0,
            misses=0,
            copied_bytes=0,
        )
        self.run_hook = self.run
        self.close_hook = self.close_neural
        n.prefill_lib.mk2_native_prefill_run = self.run_hook
        n.close = self.close_hook
        engine._fp32_prefix_snapshot = self

    def owner_identity(self):
        n = self.n
        return (
            n.prefill_handle,
            n.stage0_handle,
            n.stream.cuda_stream,
            n.library_sha256,
            n.prefill_library_sha256,
            n.final_hidden.data_ptr(),
            int(n.bindings.slot_map),
            int(n.bindings.seq_len),
            tuple(
                (
                    int(n.bindings.conv_state[i]),
                    int(n.bindings.ssm_state[i]),
                    n.bindings.conv_slot_stride[i],
                    n.bindings.ssm_slot_stride[i],
                )
                for i in range(14)
            ),
            tuple(
                (int(n.bindings.kv_block_table[i]), int(n.bindings.kv_pool_base[i]))
                for i in range(3)
            ),
        )

    def prompt_key(self):
        n = self.n
        return (
            n.inputs.prompt.data_ptr(),
            n.inputs.prompt._version,
            tuple(n.inputs.prompt.shape),
            str(n.inputs.prompt.dtype),
            n.inputs.prompt_positions,
            tuple(tuple(request.context_ids) for request in n.inputs.requests),
        )

    def prompt_layout(self):
        n = self.n
        return (
            n.inputs.prompt.data_ptr(),
            tuple(n.inputs.prompt.shape),
            tuple(n.inputs.prompt.stride()),
            n.inputs.prompt.storage_offset(),
            str(n.inputs.prompt.dtype),
            str(n.inputs.prompt.device),
            n.inputs.prompt_positions,
            n.config.batch,
            n.config.prompt_positions,
            n.config.max_sequence_tokens,
            n.config.tokens_per_block,
        )

    def byte_view(self, pointer, size):
        require(pointer and size > 0, "Invalid prefix allocation")

        class Alias:
            def __init__(self):
                self.__cuda_array_interface__ = dict(
                    shape=(size,),
                    typestr="|u1",
                    data=(int(pointer), False),
                    version=3,
                    strides=None,
                )

        return self.t.as_tensor(Alias(), device="cuda:0")

    def state_views(self):
        n = self.n
        b, p, blocks = n.batch, n.inputs.prompt_positions, n.info.blocks_per_sequence
        require(1 <= p <= blocks * 32, "Prefix exceeds allocated KV capacity")
        expected_kv = b * 2 * blocks * 4 * 32 * 128 * 2
        require(n.info.kv_pool_bytes_per_layer == expected_kv, "Unexpected KV geometry")
        views = [
            n.final_hidden,
            self.byte_view(n.bindings.slot_map, b * 4),
            self.byte_view(n.bindings.seq_len, b * 4),
        ]
        for i in range(14):
            views.extend(
                (
                    self.byte_view(n.bindings.conv_state[i], b * 10752 * 2),
                    self.byte_view(n.bindings.ssm_state[i], b * 196608 * 4),
                )
            )
        for i in range(3):
            views.append(self.byte_view(n.bindings.kv_block_table[i], b * 2 * blocks * 4))
            pool = self.byte_view(n.bindings.kv_pool_base[i], expected_kv).reshape(
                b, 2, blocks, 4, 32, 256
            )
            for block in range((p + 31) // 32):
                views.append(pool[:, :, block, :, : min(32, p - block * 32), :])
        return views

    def run(self, handle, stream):
        n = self.n
        require(not self.closed and not n.closed, "Closed prefix snapshot")
        require(
            handle == n.prefill_handle and stream == n.stream.cuda_stream,
            "Prefix call targets a different owner or stream",
        )
        require(self.owner_identity() == self.identity, "Prefix allocation ownership changed")
        require(
            self.prompt_layout() == self.prompt_allocation,
            "Native prefill prompt allocation changed; construct a new owner",
        )
        if self.prompt_key() != self.prompt_identity:
            self.receipt["misses"] += 1
            return self.original(handle, stream)
        with self.t.cuda.stream(n.stream):
            for destination, snapshot in zip(self.views, self.snapshots, strict=True):
                destination.copy_(snapshot)
        self.receipt["hits"] += 1
        self.receipt["copied_bytes"] += self.receipt["bytes"]
        return 0

    def detach(self):
        if self.closed:
            return
        n, engine = self.n, self.engine
        require(
            getattr(engine, "_fp32_prefix_snapshot", None) is self,
            "Prefix engine owner was replaced",
        )
        require(
            n.prefill_lib.mk2_native_prefill_run is self.run_hook,
            "Prefix hook was replaced by another owner",
        )
        require(n.close is self.close_hook, "Neural close hook was replaced")
        n.stream.synchronize()
        n.prefill_lib.mk2_native_prefill_run = self.original
        n.close = self.original_close
        self.snapshots.clear()
        self.views.clear()
        engine._fp32_prefix_receipt = dict(self.receipt)
        del engine._fp32_prefix_snapshot
        self.engine = None
        self.n = None
        self.original = None
        self.original_close = None
        self.run_hook = None
        self.close_hook = None
        self.t = None
        self.closed = True

    def close_neural(self):
        if self.closed:
            return
        original_close = self.original_close
        self.detach()
        original_close()


def install(engine):
    return PrefixSnapshot(engine)
