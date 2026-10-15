def abort_response(request_id, completed_ids, abort):
    if not isinstance(request_id, str):
        return "UNKNOWN_REQUEST"
    if request_id in completed_ids:
        return "KNOWN_TERMINAL"
    return "ACTIVE_ABORT_ACCEPTED" if abort(request_id) else "UNKNOWN_REQUEST"
