"""Self-contained generated circuits and waveform comparison helpers."""
import math
import bisect

def cases():
    return {
        "rc": ("Vdd in 0 pulse(0 1.8 1n 100p 100p 4n 10n)\nR1 in out 1k\nC1 out 0 1p",
               "v(in) v(out) i(vdd)", "tran 20p 20n 0 20p"),
        "controlled-rlc": ("Vdd in 0 pulse(0 1 1n 200p 200p 4n 10n) ac 1\n"
            "R1 in n1 50\nL1 n1 n2 10n\nC1 n2 0 1p\nR2 n2 0 1k\n"
            "E1 out 0 n2 0 2\nG1 out n1 n2 0 1m\nR3 out 0 500",
            "v(in) v(n2) v(out) i(vdd)", "tran 20p 20n 0 20p"),
        "diode": ("Vdd in 0 sin(0 2 100meg)\nR1 in out 100\nC1 out 0 1p\n"
            "D1 out 0 dm\n.model dm d is=1e-14 n=1.05 cjo=20f",
            "v(in) v(out) i(vdd)", "tran 10p 20n 0 10p"),
        "mos": ("Vdd vdd 0 1.8\nVin in 0 pulse(0 1.8 0.2n 50p 50p 0.8n 2n)\n"
            ".model nch nmos level=54 version=4.8.3 toxe=2.6n toxp=2.6n vth0=0.45 u0=0.04\n"
            ".model pch pmos level=54 version=4.8.3 toxe=2.6n toxp=2.6n vth0=-0.45 u0=0.02\n" +
            "\n".join(f"Mn{i} out{i} in 0 0 nch w=1u l=180n\nMp{i} out{i} in vdd vdd pch w=2u l=180n\nC{i} out{i} 0 {10+i%7}f" for i in range(64)),
            "v(in) v(out0) i(vdd)", "tran 5p 2n 0 5p"),
        "analysis-switch": ("Vdd in 0 dc 0.5 ac 1 pulse(0.5 1 1n 100p 100p 4n 10n)\n"
            "R1 in out 1k\nC1 out 0 1p\nR2 out 0 2k",
            "v(in) v(out) i(vdd)",
            "op\ntran 20p 20n 0 20p\nac dec 10 1meg 10gig\n"
            "wrdata ac.dat v(out)\ntran 20p 20n 0 20p"),
    }

def compare(reference, candidate, rtol=2e-4, atol=2e-9):
    names, expected = read_wave(reference)
    names2, actual = read_wave(candidate)
    assert names == names2
    assert abs(expected[-1][0] - actual[-1][0]) <= abs(expected[-1][0]) * 1e-10
    assert abs(expected[0][0] - actual[0][0]) <= max(abs(expected[0][0]), 1e-30) * 1e-7
    xs = [r[0] for r in actual]
    errors = [0.] * (len(names) - 1)
    for row in expected:
        j = max(1, min(len(actual) - 1, bisect.bisect_left(xs, row[0])))
        a, b = actual[j - 1], actual[j]
        alpha = (row[0] - a[0]) / (b[0] - a[0])
        for k in range(1, len(names)):
            errors[k - 1] = max(errors[k - 1], abs(row[k] - (a[k] + alpha * (b[k] - a[k]))))
    values = []
    for k, err in enumerate(errors, 1):
        scale = max(abs(row[k]) for row in expected)
        limit = atol + rtol * scale
        area = lambda rows: sum((b[0] - a[0]) * (a[k] + b[k]) / 2 for a, b in zip(rows, rows[1:]))
        integral_error = abs(area(actual) - area(expected))
        values.append({"vector": names[k], "max_abs_error": err, "limit": limit,
                       "integral_abs_error": integral_error, "reference_integral": area(expected)})
        if err > limit:
            raise AssertionError(f"{candidate}: {names[k]} error {err} exceeds {limit}")
    return {"reference_points": len(expected), "candidate_points": len(actual),
            "byte_identical": reference.read_bytes() == candidate.read_bytes(), "vectors": values}

def read_wave(path):
    lines = path.read_text().splitlines()
    names = lines[0].split()
    data = [list(map(float, row.split())) for row in lines[1:] if row.strip()]
    if len(data) < 10 or any(len(row) != len(names) for row in data):
        raise AssertionError(f"Incomplete waveform {path}")
    if any(not math.isfinite(x) for row in data for x in row):
        raise AssertionError(f"Non-finite waveform {path}")
    if any(b[0] <= a[0] for a, b in zip(data, data[1:])):
        raise AssertionError(f"Unordered waveform {path}")
    return names, data
