import json
import re
from pathlib import Path

from .bundle import BundleError


def validate_native_device(cuda, receipt_dir=None):
    report = {
        "schema": "nemotron-native-device.v1",
        "status": "FAIL",
        "device_index": 0,
        "required": {
            "family": "H100",
            "compute_capability": [9, 0],
            "sm_count": 132,
            "compiled_architecture": "sm_90a",
            "mig": False,
        },
        "scope": "Qualified device profile; kernel occupancy and allocation checks still apply",
    }
    try:
        if not cuda.is_available():
            raise BundleError("CUDA device unavailable; check the driver and GPU container access")
        properties = cuda.get_device_properties(0)
        name = str(properties.name)
        capability = [int(properties.major), int(properties.minor)]
        sms = int(properties.multi_processor_count)
        report["detected"] = {
            "name": name,
            "compute_capability": capability,
            "sm_count": sms,
            "total_memory_bytes": int(properties.total_memory),
        }
        if capability != [9, 0]:
            raise BundleError(
                "The native kernels require Hopper sm_90a; other architectures need a port"
            )
        if not re.search(r"\bH100\b", name, re.IGNORECASE) or re.search(
            r"\bMIG\b", name, re.IGNORECASE
        ):
            raise BundleError("This native release is qualified only for a full H100 device")
        if sms != 132:
            raise BundleError(
                "The native launch and synchronization profiles require 132 SMs; "
                "a smaller device or MIG partition needs separately qualified profiles"
            )
        report["status"] = "PASS"
    except Exception as error:
        detected = report.get("detected", {})
        report["error"] = (
            f"Unsupported native GPU: {detected or 'not detected'}. {error}. "
            "No model weights were loaded and no neural kernel was launched."
        )
    if receipt_dir is not None:
        path = Path(receipt_dir) / "hardware.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise BundleError(report["error"])
    return report
