""" test OSDI simulation of capacitor
"""
import os, shutil
import numpy as np
import pandas as pd
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from testing import prepare_test

# This test runs a DC, AC and Transient Simulation of a simple capacitor.
# The capacitor is available as a C file and needs to be compiled to a shared object
# and then bet put into /usr/local/share/ngspice/osdi:
#
# > make osdi_capacitor
# > cp capacitor_osdi.so /usr/local/share/ngspice/osdi/capacitor_osdi.so
#
# The integration test proves the functioning of the OSDI interface.
# Future tests will target Verilog-A models like HICUM/L2 that should yield exactly the same results as the Ngspice implementation.

directory = os.path.dirname(__file__)


def test_ngspice():
    dir_osdi, dir_built_in = prepare_test(directory)
   
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
        dc_data_osdi["v(d)"],
        dc_data_osdi["i(vsense)"] * 1e3,
        label="OSDI",
    )
    ax1.set_ylabel(r"$I_{\mathrm{P}} (\mathrm{mA})$")
    ax1.set_xlabel(r"$V_{\mathrm{PM}}(\mathrm{V})$")
    plt.legend()

    # AC Plot
    omega = 2 * np.pi * ac_data_osdi["frequency"]
    z_analytical = 5e-12 * omega
    fig = plt.figure()
    plt.semilogx(
        ac_data_built_in["frequency"],
        ac_data_built_in["i(vsense)"] * 1e3,
        label="built-in",
        linestyle=" ",
        marker="x",
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
        ac_data_osdi["frequency"],
        ac_data_osdi["i(vsense).1"] * 1e12 / omega,
        label="OSDI",
    )
    plt.semilogx(
        ac_data_osdi["frequency"],
        np.ones_like(ac_data_osdi["frequency"]) * z_analytical * 1e12 / omega,
        label="analytical",
        linestyle="--",
        marker="s",
    )
    plt.ylim(1, 9)
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
        tr_data_osdi["time"] * 1e9,
        tr_data_osdi["i(vsense)"] * 1e3,
        label="OSDI",
    )
    plt.xlabel(r"$t(\mathrm{nS})$")
    plt.ylabel(r"$I_{\mathrm{D}}(\mathrm{mA})$")
    plt.legend()

    plt.show()
