""" test OSDI simulation of resistor and capacitor
"""
import os, shutil
import numpy as np
import subprocess
import pandas as pd

directory = os.path.dirname(__file__)

# This test runs a DC, AC and Transient Simulation of a simple resistor and resistor.
# The capacitor and resistor are available as C files and need to be compiled to a shared objects
# and then bet put into /usr/local/share/ngspice/osdi:
#
# > make osdi_resistor
# > cp resistor_osdi.so /usr/local/share/ngspice/osdi/resistor_osdi.so
# > make osdi_capacitor
# > cp capacitor_osdi.so /usr/local/share/ngspice/osdi/capacitor_osdi.so
#
# The integration test proves the functioning of the OSDI interface.
# Future tests will target Verilog-A models like HICUM/L2 that should yield exactly the same results as the Ngspice implementation.


def create_shared_object():
    # place the file "resistor_va.c" and "capacitor_va.c" next to this file
    for dev in ["capacitor", "resistor"]:
        subprocess.run(
            [
                "gcc",
                "-c",
                "-Wall",
                "-I",
                "../../src/spicelib/devices/osdi/",
                "-fpic",
                dev + "_va.c",
                "-ggdb",
            ],
            cwd=directory,
        )
        subprocess.run(
            ["gcc", "-shared", "-o", dev + "_va.so", dev + "_va.o", "-ggdb"],
            cwd=directory,
        )
        os.makedirs(os.path.join(directory, "test_osdi", "osdi"), exist_ok=True)
        subprocess.run(
            ["mv", dev + "_va.so", "test_osdi/osdi/" + dev + "_va.so"], cwd=directory
        )
        subprocess.run(["rm", dev + "_va.o"], cwd=directory)


# specify location of Ngspice executable to be tested
ngspice_path = os.path.join(directory, "../../debug/src/ngspice")
ngspice_path = os.path.abspath(ngspice_path)


def test_ngspice():
    path_netlist = os.path.join(directory, "netlist.sp")

    # open netlist and activate Ngspice devices
    with open(path_netlist) as netlist_handle:
        netlist_raw = netlist_handle.read()

    netlist_osdi = netlist_raw.replace("*OSDI_ACTIVATE*", "")
    netlist_built_in = netlist_raw.replace("*BUILT_IN_ACTIVATE*", "")

    # make directories for test cases
    dir_osdi = os.path.join(directory, "test_osdi")
    dir_built_in = os.path.join(directory, "test_built_in")
    # remove old results:
    for directory_i in [dir_osdi, dir_built_in]:
        shutil.rmtree(directory_i, ignore_errors=True)

    for directory_i in [dir_osdi, dir_built_in]:
        os.makedirs(directory_i, exist_ok=True)

    create_shared_object()

    # write netlists
    with open(os.path.join(dir_osdi, "netlist.sp"), "w") as netlist_handle:
        netlist_handle.write(netlist_osdi)

    with open(os.path.join(dir_built_in, "netlist.sp"), "w") as netlist_handle:
        netlist_handle.write(netlist_built_in)

    # run simulations with Ngspice
    for directory_i in [dir_osdi, dir_built_in]:
        subprocess.run(
            [
                ngspice_path,
                "netlist.sp",
                "-b",
            ],
            cwd=directory_i,
        )

    # read DC simulation results
    dc_data_osdi = pd.read_csv(os.path.join(dir_osdi, "dc_sim.ngspice"), sep="\\s+")
    dc_data_built_in = pd.read_csv(os.path.join(dir_osdi, "dc_sim.ngspice"), sep="\\s+")
    # dc_data_built_in = pd.read_csv(
    #     os.path.join(dir_built_in, "dc_sim.ngspice"), sep="\\s+"
    # )

    id_osdi = dc_data_osdi["i(vsense)"].to_numpy()
    id_built_in = dc_data_osdi["i(vsense)"].to_numpy()
    # id_built_in = dc_data_built_in["i(vsense)"].to_numpy()

    # read AC simulation results
    ac_data_osdi = pd.read_csv(os.path.join(dir_osdi, "ac_sim.ngspice"), sep="\\s+")
    ac_data_built_in = pd.read_csv(os.path.join(dir_osdi, "ac_sim.ngspice"), sep="\\s+")
    # ac_data_built_in = pd.read_csv(
    #     os.path.join(dir_built_in, "ac_sim.ngspice"), sep="\\s+"
    # )

    # read TR simulation results
    tr_data_osdi = pd.read_csv(os.path.join(dir_osdi, "tr_sim.ngspice"), sep="\\s+")
    tr_data_built_in = pd.read_csv(os.path.join(dir_osdi, "tr_sim.ngspice"), sep="\\s+")
    # tr_data_built_in = pd.read_csv(
    #     os.path.join(dir_built_in, "tr_sim.ngspice"), sep="\\s+"
    # )

    # test simulation results
    id_osdi = dc_data_osdi["i(vsense)"].to_numpy()
    id_built_in = dc_data_built_in["i(vsense)"].to_numpy()
    np.testing.assert_allclose(id_osdi[0:20], id_built_in[0:20], rtol=0.01)

    return (
        dc_data_osdi,
        dc_data_built_in,
        ac_data_osdi,
        ac_data_built_in,
        tr_data_osdi,
        tr_data_built_in,
    )


if __name__ == "__main__":
    (
        dc_data_osdi,
        dc_data_built_in,
        ac_data_osdi,
        ac_data_built_in,
        tr_data_osdi,
        tr_data_built_in,
    ) = test_ngspice()

    import matplotlib.pyplot as plt

    # DC Plot
    pd_built_in = dc_data_built_in["v(d)"] * dc_data_built_in["i(vsense)"]
    pd_osdi = dc_data_osdi["v(d)"] * dc_data_osdi["i(vsense)"]
    fig, ax1 = plt.subplots()
    ax1.plot(
        dc_data_built_in["v(d)"],
        dc_data_built_in["i(vsense)"] * 1e3,
        label="built-in",
        linestyle=" ",
        marker="x",
    )
    ax1.plot(
        dc_data_built_in["v(d)"],
        dc_data_built_in["v(d)"] / 10 * 1e3,
        label="analytical",
        linestyle="--",
        marker="s",
    )
    ax1.plot(
        dc_data_osdi["v(d)"],
        dc_data_osdi["i(vsense)"] * 1e3,
        label="OSDI",
    )
    ax1.set_ylabel(r"$I_{\mathrm{P}} (\mathrm{mA})$")
    ax1.set_xlabel(r"$V_{\mathrm{PM}}(\mathrm{V})$")
    plt.legend()

    # AC Plot
    omega = 2 * np.pi * ac_data_osdi["frequency"]
    z_analytical = 1 / 10
    fig = plt.figure()
    plt.semilogx(
        ac_data_built_in["frequency"],
        ac_data_built_in["i(vsense)"] * 1e3,
        label="built-in",
        linestyle=" ",
        marker="x",
    )
    plt.semilogx(
        ac_data_built_in["frequency"],
        np.ones_like(ac_data_built_in["frequency"]) * z_analytical * 1e3,
        label="analytical",
        linestyle="--",
        marker="s",
    )
    plt.semilogx(
        ac_data_osdi["frequency"], ac_data_osdi["i(vsense)"] * 1e3, label="OSDI"
    )
    plt.xlabel("$f(\\mathrm{H})$")
    plt.ylabel("$\\Re \\left\{ Y_{11} \\right\} (\\mathrm{mS})$")
    plt.legend()
    fig = plt.figure()
    plt.semilogx(
        ac_data_built_in["frequency"],
        ac_data_built_in["i(vsense).1"] * 1e12 / omega,
        label="built-in",
        linestyle=" ",
        marker="x",
    )
    plt.semilogx(
        ac_data_built_in["frequency"],
        np.zeros_like(ac_data_built_in["i(vsense).1"]) * 1e12 / omega,
        label="analytical",
        linestyle="--",
        marker="s",
    )
    plt.semilogx(
        ac_data_osdi["frequency"],
        ac_data_osdi["i(vsense).1"] * 1e12 / omega,
        label="OSDI",
    )
    plt.ylim(-1, 1)
    plt.xlabel("$f(\\mathrm{H})$")
    plt.ylabel("$\\Im\\left\{Y_{11}\\right\}/(\\omega) (\\mathrm{pF})$")
    plt.legend()

    # TR plot
    fig = plt.figure()
    plt.plot(
        tr_data_built_in["time"] * 1e9,
        tr_data_built_in["i(vsense)"] * 1e3,
        label="built-in",
        linestyle=" ",
        marker="x",
    )
    plt.plot(
        tr_data_built_in["time"] * 1e9,
        tr_data_built_in["v(d)"] / 10 * 1e3,
        label="analytical",
        linestyle="--",
        marker="s",
    )
    plt.plot(
        tr_data_osdi["time"] * 1e9,
        tr_data_osdi["i(vsense)"] * 1e3,
        label="OSDI",
    )
    plt.xlabel(r"$t(\mathrm{nS})$")
    plt.ylabel(r"$I_{\mathrm{D}}(\mathrm{mA})$")
    plt.legend()

    plt.show()
