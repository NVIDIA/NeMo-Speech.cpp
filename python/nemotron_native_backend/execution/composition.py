from __future__ import annotations

import copy
import ctypes as C
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ..bundle import BundleError, NativeBundlePlan, resolve_artifacts
from . import (
    chunk_submission,
    core,
    helper_graphs,
    pcm_transfer,
    prefix_cache,
    state_binding,
    token_staging,
)
from .engine import NativeCohortEngine
from .lane_control import update_first_eos
from .native_assets import ModelAssets
from .native_inputs import requests_from_texts
from .source_build import is_source_build, verify_source_build


class PackageSupport:
    ModelAssets = ModelAssets
    Cohort = core.Cohort
    requests_from_texts = staticmethod(requests_from_texts)
    update_first_eos = staticmethod(update_first_eos)

    def __init__(self, plan, artifacts, source_pins):
        self.plan = plan
        self.artifacts = artifacts
        self.source_pins = dict(source_pins)
        self.source_sha256 = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()

    def verify_sources(self):
        package = Path(__file__).resolve().parent.parent
        if not self.source_pins:
            raise BundleError("Frozen package source pins are required")
        actual = {str(path.relative_to(package)) for path in package.rglob("*.py")}
        if set(self.source_pins) != actual:
            raise BundleError("Package source inventory differs from the qualification seal")
        for relative, expected in self.source_pins.items():
            target = (package / relative).resolve(strict=True)
            if (
                not target.is_relative_to(package)
                or hashlib.sha256(target.read_bytes()).hexdigest() != expected
            ):
                raise BundleError("Package source differs from qualification seal: " + relative)

    def select(self, batch):
        if batch != self.plan.physical_width:
            raise BundleError("Requested physical width differs from the kernel binding")
        paths = {
            name: self.artifacts[key]
            for name, key in (("prefill", "prefill"), ("stage0", "stage0"), ("stage1", "codec"))
        }
        return paths, dict(
            physical_batch=batch,
            variant=self.plan.identity,
            neural_cohort_partitioning=False,
            bundle_manifest_sha256=self.plan.manifest_sha256,
        )


def verify_compiled_contract(plan, artifacts):
    value = plan.contract
    built = verify_source_build(plan, artifacts) if is_source_build(value) else None
    binding = (
        built["compiled_binding"]
        if built is not None
        else json.loads(artifacts["compiled_binding"].read_text())
    )
    if (
        binding["batch"] != plan.physical_width
        or binding["physical_batch"] != plan.physical_width
        or binding["neural_cohort_partitioning"] is not False
        or binding["state_mode"] != value["state"]["mode"]
        or binding["state_format"] != value["state"]["format"]
        or binding["ctas"] != value["execution"]["stage0_grid"]
        or binding["prefill_workspace_sequences"]
        != value["kernels"]["prefill"]["workspace_sequences"]
    ):
        raise BundleError("Compiled binding does not satisfy the proposed program")
    expected_options = {
        "binding_module": "state_r9.binding",
        "batched_handoff": True,
        "pinned_pcm": True,
        "early_proof": True,
        "engine_binding": "integration_r9.binding",
        "engine_bindings": ["control_r9.chunk", "control_r9.prefix"],
        "transport_binding": "publication.wide_delivery",
    }
    if value["kernels"]["prefill"].get("validation_kind") == "retained-device-dedup":
        expected_options.pop("transport_binding")
    if binding["runtime_options"] != expected_options:
        raise BundleError("Compiled binding requires unported construction hooks")
    if built is None:
        for region in ("prefill", "stage0", "codec"):
            receipt = json.loads(artifacts[region + "_build_receipt"].read_text())
            if (
                receipt["library_sha256"]
                != hashlib.sha256(artifacts[region].read_bytes()).hexdigest()
            ):
                raise BundleError("Library and immutable build receipt disagree: " + region)
    recipe = json.loads(artifacts["state_recipe"].read_text())
    if (
        recipe["mode"] != value["state"]["mode"]
        or recipe["reordering"] != value["state"]["map_kind"]
    ):
        raise BundleError("State recipe and compiled state mode disagree")
    for field, key in (("scales", "state_scales"), ("map", "state_map")):
        if recipe[field]["sha256"] != hashlib.sha256(artifacts[key].read_bytes()).hexdigest():
            raise BundleError("State artifact differs from its recipe: " + field)
    return binding


def bind_state_and_startup(neural, plan, artifacts):
    function = neural.lib.mk2_set_native_predictor_active_mask
    function.argtypes = [C.c_void_p, C.c_void_p]
    function.restype = C.c_int
    neural._check(
        function(neural.stage0_handle, neural.inputs.audio_valid.data_ptr()), "startup mask"
    )
    neural._startup_skip_mask_owner = neural.inputs.audio_valid
    state_bytes = neural.lib.mk2_r9_state_bytes
    state_bytes.argtypes = []
    state_bytes.restype = C.c_int
    if state_bytes() != 2:
        raise BundleError("Compiled recurrent state is not BF16 private storage")
    state_binding.install(
        neural,
        state_mode=plan.contract["state"]["mode"],
        scales_path=artifacts["state_scales"],
        map_path=artifacts["state_map"],
        recipe_identity=artifacts["state_recipe"],
    )
    if plan.contract["kernels"]["prefill"].get("validation_kind") == "retained-device-dedup":
        if neural.batch != 512:
            raise BundleError("Legacy deduplicated prefill is qualified only at B512")
        return
    original = neural.prefill_lib.mk2_native_prefill_run
    validate = neural.prefill_lib.mk2_native_prefill_validate
    validate.argtypes = [C.c_void_p, C.c_void_p]
    validate.restype = C.c_int

    def checked_prefill(handle, stream):
        status = original(handle, stream)
        return status if status else validate(handle, stream)

    neural.prefill_lib.mk2_native_prefill_run = checked_prefill
    neural._r14_prefill_validation = dict(
        entry="mk2_native_prefill_validate",
        scope="cache preparation and every miss before consumers or snapshots",
        workspace_sequences=1,
    )


def bind_receipt_contract(engine, plan, binding):
    original = engine.stream
    admission = dict(
        initial_singleton_readiness=plan.contract["publication"]["initial_singleton_readiness"],
        post_readiness_width=plan.physical_width,
        incomplete_deadline_ms=plan.contract["publication"]["incomplete_deadline_ms"],
    )

    def stream(*args, **kwargs):
        iterator = original(*args, **kwargs)
        try:
            for event in iterator:
                if event["type"] == "receipt":
                    receipt = event["receipt"]
                    receipt["r9_configuration"] = dict(
                        id=binding["id"],
                        components=binding["r9_components"],
                        state_format=plan.contract["state"]["format"],
                        transport_format="pcm_f32_le",
                    )
                    receipt["immutable_prefix_cache"] = copy.deepcopy(engine._r9_prefix_cache[3])
                    receipt["r21_admission"] = dict(admission)
                    receipt["native_package"] = dict(
                        id=plan.identity,
                        manifest_sha256=plan.manifest_sha256,
                        qualification_only=True,
                    )
                yield event
        finally:
            iterator.close()

    engine.stream = stream


class ConfiguredEngine(NativeCohortEngine):
    def __init__(
        self,
        plan,
        artifacts,
        artifact_root,
        model_root,
        source_pins,
        binding,
        *,
        research_schedule=(3, 5),
        research_steps=None,
    ):
        self.native_plan = plan
        self.native_artifacts = artifacts
        self.native_compiled_binding = binding
        support = PackageSupport(plan, artifacts, source_pins)
        super().__init__(
            plan.physical_width,
            model_root=model_root,
            steps=research_steps
            if research_steps is not None
            else plan.contract["generation"]["cap"],
            retain_codes=False,
            support=support,
            artifact_root=artifact_root,
            qualification_mode=True,
            execution=dict(plan.contract["execution"]),
            active_lanes={"mode": "disabled"},
            research_schedule=research_schedule,
        )
        try:
            expected = plan.contract["execution"]
            if (
                self.residency["stage0"]["batch"] != plan.physical_width
                or self.residency["stage0"]["grid"] != expected["full_stage0_grid"]
                or self.residency["codec_blocks"] != expected["codec_grid"]
            ):
                raise BundleError(
                    "Actual launch does not match the physical width and resource program"
                )
            geometry = plan.contract["kernels"]["stage0"]
            actual = self.residency["stage0"]
            for declared, observed in (
                ("row_tile", "batch_pad"),
                ("k_tile", "kchunk"),
                ("pipeline_stages", "stages"),
            ):
                if geometry[declared] != actual[observed]:
                    raise BundleError(
                        "Declared producer geometry differs from compiled ABI: " + declared
                    )
            token_staging.install(self, mode=plan.contract["host"]["tokens"])
        except BaseException:
            self.close()
            raise

    def _allocate_queues(self):
        bind_state_and_startup(self.core.neural, self.native_plan, self.native_artifacts)
        self.startup_predictor_skip = True
        self.helper_graphs = helper_graphs.install(self.core.neural)
        super()._allocate_queues()
        self.r8_options = dict(self.native_compiled_binding["runtime_options"])
        chunk_submission.install(self)
        prefix_cache.install(self)
        self._r9_prefix_cache[3]["key"]["state"] = self.native_plan.contract["state"]["format"]
        bind_receipt_contract(self, self.native_plan, self.native_compiled_binding)
        pcm_transfer.install(self)


def create_for_qualification(
    plan: NativeBundlePlan,
    artifact_root: Path,
    model_root: Path,
    *,
    source_pins: Mapping[str, str],
    model_identity: str,
    qualification_only: bool,
):
    if qualification_only is not True or plan.contract["enabled"] is not False:
        raise BundleError(
            "The native port is available only through an explicit qualification call"
        )
    if model_identity != plan.contract["model"]["checkpoint_sha256"]:
        raise BundleError("Caller model identity differs from the frozen proposal")
    if plan.physical_width not in (128, 256, 512, 1024, 2048, 4096):
        raise BundleError("Unported wide construction contract")
    artifacts = resolve_artifacts(plan, artifact_root)
    binding = verify_compiled_contract(plan, artifacts)
    return ConfiguredEngine(
        plan, artifacts, Path(artifact_root), Path(model_root), source_pins, binding
    )
