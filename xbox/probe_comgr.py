# ctypes driver over System32 amd_comgr_3.dll. Two jobs:
#   python xbox/probe_comgr.py            # enumerate ISAs this COMGR knows
#   python xbox/probe_comgr.py <target>   # enumerate + compile <target>.hip probes
# This is the ONLY compile path left on this box: hip/rtc_compile.cpp cannot be
# recompiled (cc1plus aborts with NTSTATUS 0xC0000135 STATUS_ACCESS_DENIED as soon
# as it writes an output file), and hip/rtc_compile.exe is the 16.10.2026 binary,
# which hard-throws on any gfx115x target.
# COMPILE-ONLY: no execution, so this does NOT prove device behaviour.
import ctypes, os, re, sys

BOX = os.path.dirname(os.path.abspath(__file__))
DLL = 'amd_comgr_3.dll'
LANG_HIP = 3
KIND_SOURCE, KIND_DIAGNOSTICS, KIND_LOG, KIND_EXEC = 1, 4, 5, 8
# Action ids and data kinds are separate enums here: rtc_compile.cpp reads the
# assembly out of the assembly data set at data kind 1, NOT at action id 5.
KIND_ASSEMBLY = 1
ACT_COMPILE, ACT_ASSEMBLY, ACT_RELOC, ACT_EXEC = 2, 5, 4, 7


def load():
    c = ctypes.WinDLL(DLL)
    c.amd_comgr_get_version.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    c.amd_comgr_get_isa_count.argtypes = [ctypes.c_void_p]
    c.amd_comgr_get_isa_name.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]
    c.amd_comgr_create_data.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    c.amd_comgr_set_data.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    c.amd_comgr_set_data_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    c.amd_comgr_create_data_set.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    c.amd_comgr_data_set_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    c.amd_comgr_create_action_info.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    c.amd_comgr_action_info_set_language.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    c.amd_comgr_action_info_set_isa_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    c.amd_comgr_action_info_set_option_list.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    c.amd_comgr_action_info_set_logging.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    c.amd_comgr_do_action.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    c.amd_comgr_action_data_count.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_size_t)]
    c.amd_comgr_action_data_get_data.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    c.amd_comgr_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    return c


def _text(c, out_set, act, lines):
    n = ctypes.c_size_t()
    if c.amd_comgr_action_data_count(out_set, act, ctypes.byref(n)) != 0:
        return
    for i in range(n.value):
        d = ctypes.c_void_p()
        if c.amd_comgr_action_data_get_data(out_set, act, i, ctypes.byref(d)) != 0:
            continue
        sz = ctypes.c_size_t()
        if c.amd_comgr_get_data(d, ctypes.byref(sz), None) != 0:
            continue
        buf = ctypes.create_string_buffer(sz.value + 1)
        if c.amd_comgr_get_data(d, ctypes.byref(sz), buf) != 0:
            continue
        lines.append(buf.value.decode('utf-8', 'replace'))


def _bytes(c, out_set, act, idx):
    d = ctypes.c_void_p()
    if c.amd_comgr_action_data_get_data(out_set, act, idx, ctypes.byref(d)) != 0:
        return None
    sz = ctypes.c_size_t()
    if c.amd_comgr_get_data(d, ctypes.byref(sz), None) != 0:
        return None
    buf = ctypes.create_string_buffer(sz.value + 1)
    if c.amd_comgr_get_data(d, ctypes.byref(sz), buf) != 0:
        return None
    return buf.raw[:sz.value]


def list_isas(c):
    maj, mino = ctypes.c_size_t(), ctypes.c_size_t()
    c.amd_comgr_get_version(ctypes.byref(maj), ctypes.byref(mino))
    cnt = ctypes.c_size_t()
    if c.amd_comgr_get_isa_count(ctypes.byref(cnt)) != 0:
        print("COMGR get_isa_count failed")
        return [], None
    names = []
    for i in range(cnt.value):
        name = ctypes.c_char_p()
        if c.amd_comgr_get_isa_name(i, ctypes.byref(name)) == 0 and name.value:
            names.append(name.value.decode())
    print("COMGR %d.%d  ISAs=%d" % (maj.value, mino.value, len(names)))
    return names, (maj.value, mino.value)


def compile_hip(c, src_text, name, target, extra_opts=()):
    """Mirrors hip/rtc_compile.cpp's action sequence. Returns (rc, log_text, hsaco, assembly)."""
    lines = []
    src = ctypes.c_void_p()
    if c.amd_comgr_create_data(KIND_SOURCE, ctypes.byref(src)) != 0:
        return 1, "create_data failed", None, None
    blob = src_text.encode()
    c.amd_comgr_set_data(src, len(blob), blob)
    c.amd_comgr_set_data_name(src, name.encode())
    inp, out = ctypes.c_void_p(), ctypes.c_void_p()
    c.amd_comgr_create_data_set(ctypes.byref(inp))
    c.amd_comgr_create_data_set(ctypes.byref(out))
    c.amd_comgr_data_set_add(inp, src)
    act = ctypes.c_void_p()
    if c.amd_comgr_create_action_info(ctypes.byref(act)) != 0:
        return 1, "create_action_info failed", None, None
    c.amd_comgr_action_info_set_language(act, LANG_HIP)
    c.amd_comgr_action_info_set_isa_name(act, ("amdgcn-amd-amdhsa--" + target).encode())
    c.amd_comgr_action_info_set_logging(act, True)
    opts = ["-O3", "-nogpuinc", "-nogpulib"] + list(extra_opts)
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    c.amd_comgr_action_info_set_option_list(act, arr, len(opts))
    status = c.amd_comgr_do_action(ACT_COMPILE, act, inp, out)
    assembly = None
    if status == 0:
        asm_set = ctypes.c_void_p()
        c.amd_comgr_create_data_set(ctypes.byref(asm_set))
        backend = ["-O3"] + [e for e in extra_opts
                             if e.startswith("-mllvm") or e.startswith("-m") or not e.startswith("-")]
        b = (ctypes.c_char_p * len(backend))(*[o.encode() for o in backend])
        c.amd_comgr_action_info_set_option_list(act, b, len(backend))
        if c.amd_comgr_do_action(ACT_ASSEMBLY, act, out, asm_set) == 0:
            n = ctypes.c_size_t()
            c.amd_comgr_action_data_count(asm_set, KIND_ASSEMBLY, ctypes.byref(n))
            if n.value:
                assembly = _bytes(c, asm_set, KIND_ASSEMBLY, 0)
        reloc = ctypes.c_void_p()
        c.amd_comgr_create_data_set(ctypes.byref(reloc))
        status = c.amd_comgr_do_action(ACT_RELOC, act, out, reloc)
        out = reloc
        if status == 0:
            exec_set = ctypes.c_void_p()
            c.amd_comgr_create_data_set(ctypes.byref(exec_set))
            c.amd_comgr_action_info_set_option_list(act, None, 0)
            status = c.amd_comgr_do_action(ACT_EXEC, act, reloc, exec_set)
            out = exec_set
    for kind in (KIND_DIAGNOSTICS, KIND_LOG):
        _text(c, out, kind, lines)
    hsaco = None
    if status == 0:
        n = ctypes.c_size_t()
        if c.amd_comgr_action_data_count(out, KIND_EXEC, ctypes.byref(n)) == 0 and n.value == 1:
            hsaco = _bytes(c, out, KIND_EXEC, 0)
        else:
            status = 1
            lines.append("expected one hsaco")
    return status, "\n".join(lines), hsaco, assembly


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    c = load()
    names, ver = list_isas(c)
    if target is None:
        for n in names:
            print("  isa = %s" % n)
        print("HAS gfx115x:", any('gfx115' in n for n in names))
        return
    isa = "amdgcn-amd-amdhsa--" + target
    if isa not in names:
        print("COMGR lacks %s" % target)
        return
    path = os.path.join(BOX, target + '.hip')
    try:
        src = open(path, 'r', errors='replace').read()
    except OSError:
        print("no source %s" % path)
        return
    status, log, hsaco, assembly = compile_hip(c, src, target + ".hip", target)
    print("%s rc=%d hsaco=%s asm=%s" % (target, status,
          ("%d B" % len(hsaco)) if hsaco else "none",
          ("%d B" % len(assembly)) if assembly else "none"))
    for ln in log.splitlines():
        m = re.search(r"error: '([^']*)' needs target feature ([^\n]*)", ln)
        if m:
            print("  needs [%s] for %s" % (m.group(2).strip(), m.group(1)))
        elif "error" in ln:
            print("  " + ln.strip()[:160])
    if hsaco and len(hsaco) > 4 and hsaco[:4] == b"\x7fELF":
        open(os.path.join(BOX, target + ".hsaco"), "wb").write(hsaco)
        print("  hsaco saved %d B" % len(hsaco))
    if assembly:
        open(os.path.join(BOX, target + ".hsaco.s"), "wb").write(assembly)
        print("  assembly saved %d B" % len(assembly))


def cli_compile(argv):
    # Drop-in stand-in for hip/rtc_compile.exe, which cannot be rebuilt here and hard-rejects
    # gfx115x. Argument positions match the exe exactly, with --compile inserted:
    #   python xbox/probe_comgr.py --compile <out.hsaco> <src.hip> comgr <target>
    # RTC_EXTRA_OPTS is honoured the same way (space-separated, set per module by the build script).
    out_path, src_path = argv[2], argv[3]
    target = argv[5] if argv[4] == 'comgr' else argv[4]
    extra = list(argv[6:]) + os.environ.get('RTC_EXTRA_OPTS', '').split()
    src = open(src_path, 'r', errors='replace').read()
    status, log, hsaco, assembly = compile_hip(load(), src, 'probe.hip', target, tuple(extra))
    if log:
        print(log)
    if hsaco:
        open(out_path, 'wb').write(hsaco)
        # A whole 41-module gfx115x build would dump ~0.5 GB of assembly; RTC_NO_ASM=1 skips it.
        if assembly and not os.environ.get('RTC_NO_ASM'):
            open(out_path + '.s', 'wb').write(assembly)
        print("%s <- %s [%s] OK hsaco=%d B%s" % (out_path, src_path, target, len(hsaco),
              ' assembly emitted' if assembly and not os.environ.get('RTC_NO_ASM') else ' asm skipped'))
    sys.exit(0 if status == 0 and hsaco else 1)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--compile']:
        cli_compile(sys.argv)
    main()
