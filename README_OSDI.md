# OSDI implementation for NGSPICE

OSDI (Open Source Device Interface) is a simulator independent device interface, that is used by the OpenVAF compiler.
Implementing this interface in NGSPICE allows loading Verilog-A models compiled by OpenVAF.
The interface is fixed and does not require the compiler to know about NGSPICE during compilation.
NGSPICE also doesn't need to know anything about the compiled models at compilation.
Therefore, these models can be loaded dynamically at runtime.

To that end the `osdi` command is provided.
It allows loading a dynamic library conforming to OSDI.
Example usage: `osdi diode.osdi`.

If used within a netlist the command requires the `pre_` prefix.
This ensures that the devices are loaded before the netlist is parsed.

Example usage: `pre_osdi diode.osdi`

If a relative path is provided to the `osdi` command in a netlist, it will resolve that path **relative to the netlist**, not relative to current working directory.
This ensures that netlists can be simulated from any directory

## Binding model levels to OSDI modules

Model cards of commercial simulators select a compact model by its level
(e.g. `.model nch nmos level=72` for BSIM-CMG). The command `osdi_bind`
lets such cards use a Verilog-A module loaded with `osdi`, and lets `M`
instances refer to them:

    osdi_bind mos <level> <module> [<par> <nval> <pval>] [ignore <par> ...]
    osdi_bind mos <level> none        remove the binding
    osdi_bind                         list the bindings

Example for `.spiceinit`:

    osdi bsimcmg.osdi
    osdi_bind mos 72 bsimcmg_va ignore devtype version

`nmos` sets the polarity parameter `<par>` to `<nval>`, `pmos` to `<pval>`;
the default is `type 1 -1`. A contradicting `type` on the card is reported
and ignored. Card parameters in the `ignore` list are skipped silently.
A bound level takes precedence over a built-in model of the same level.
The module is looked up when the card is parsed, so `osdi` and `osdi_bind`
may be given in any order. Within a netlist use `pre_osdi_bind`.

## Build Instructions

To compile NGSPICE with OSDI support ensure that the `--enable-predictor` and `--enable-osdi` flags are used.
The `compile_linux.sh` file enables these flags by default.



