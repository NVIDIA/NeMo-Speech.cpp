def update_first_eos(first_eos, stop_rows, start):
    if len(first_eos) != len(stop_rows) or start < 0:
        raise ValueError("invalid EOS observation geometry")
    updated = list(first_eos)
    for lane, row in enumerate(stop_rows):
        if any(type(x) is not bool for x in row):
            raise ValueError("EOS observations must be boolean")
        if updated[lane] is None:
            updated[lane] = next((start + j for (j, stop) in enumerate(row) if stop), None)
    return updated


def completion_summary(first_eos, executed):
    if (
        executed < 0
        or not first_eos
        or any(g is not None and (not 0 <= g < executed) for g in first_eos)
    ):
        raise ValueError("EOS generation lies outside executed work")
    rows = [
        {
            "lane": i,
            "first_eos_generation": g,
            "completed": g is not None,
            "post_eos_computed_lane_steps": 0 if g is None else executed - g - 1,
        }
        for (i, g) in enumerate(first_eos)
    ]
    all_done = all(x["completed"] for x in rows)
    return {
        "all_lanes_completed": all_done,
        "completed_lanes": sum(x["completed"] for x in rows),
        "cohort_completion_generation": max(first_eos) if all_done else None,
        "request_completion": rows,
        "executed_lane_steps": len(rows) * executed,
        "discarded_post_eos_lane_steps": sum(x["post_eos_computed_lane_steps"] for x in rows),
    }
