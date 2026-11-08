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

    osdi_bind mos <level> <module> [if <cpar>] [<par> <nval> <pval>] [ignore <par> ...]
    osdi_bind dio <level> <module> [if <cpar>] [lead <ipar>] [ignore <par> ...]
    osdi_bind mos|dio <level> none [if <cpar>]   remove the binding(s)
    osdi_bind                                    list the bindings

Example for `.spiceinit`:

    osdi bsimcmg.osdi
    osdi_bind mos 72 bsimcmg_va ignore devtype version

`nmos` sets the polarity parameter `<par>` to `<nval>`, `pmos` to `<pval>`;
the default is `type 1 -1`. A contradicting `type` on the card is reported
and ignored. Card parameters in the `ignore` list are skipped silently.
A level may have several bindings. One with `if <cpar>` is selected if the
card has parameter `<cpar>` (any value), else the one without `if`. Example
for the two PSP103 modules, only the NQS module knows `swnqs`:

    osdi psp103.osdi
    osdi psp103_nqs.osdi
    osdi_bind mos 69 psp103va
    osdi_bind mos 69 pspnqs103va if swnqs

`osdi_bind mos <level> none` removes all bindings of the level,
`osdi_bind mos <level> none if <cpar>` only the one with that condition.
Diode cards `.model <name> d level=<level>` are bound with kind `dio` and used
by `D` instances. Diodes have no polarity parameter. An unlabeled value on a
`D` instance sets the instance parameter `<ipar>` given with `lead`, default
`area`:

    osdi juncap200.osdi
    osdi diode_cmc.osdi
    osdi_bind dio 6 juncap200 lead ab
    osdi_bind dio 7 diode_cmc

On a bound card, `level` selects the binding and is not passed to the module.
A bound level takes precedence over a built-in model of the same level.
The module is looked up when the card is parsed, so `osdi` and `osdi_bind`
may be given in any order. Within a netlist use `pre_osdi_bind`.

## Build Instructions

To compile NGSPICE with OSDI support ensure that the `--enable-predictor` and `--enable-osdi` flags are used.
The `compile_linux.sh` file enables these flags by default.



