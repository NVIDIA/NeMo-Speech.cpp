def install(n, *, allow_legacy_full_entry=False):
    full = n.lib.mk2_run_native_stage0
    try:
        startup = n.lib.mk2_run_native_startup_stage0
    except AttributeError:
        if allow_legacy_full_entry:
            return
        raise
    startup.argtypes = full.argtypes
    startup.restype = full.restype
    n._r8_full_stage0 = full
    n._r8_startup_stage0 = startup
    n._r8_startup_launches = 0
    n._r8_full_launches = 0

    def enqueue(*args):
        use_startup = getattr(n, "_r8_force_startup", n.inputs.generation < 5)
        if use_startup:
            n._r8_startup_launches += 1
            return startup(*args)
        n._r8_full_launches += 1
        return full(*args)

    n.lib.mk2_run_native_stage0 = enqueue
