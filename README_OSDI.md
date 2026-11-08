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
lets such cards use a Verilog-A module loaded with `osdi`, used by `M`
(MOS), `Q` (bipolar) and `D` (diode) instances:

    osdi_bind mos|bjt|dio <level> <module> [has=<par>] [area=<ipar>]
                          [polarity=<par>|none] [ignore=<par>,<par>,...]
    osdi_bind mos|bjt|dio <level> none [has=<par>]   remove the binding(s)
    osdi_bind                                        list the bindings

Example for `.spiceinit`:

    osdi psp103.osdi
    osdi psp103_nqs.osdi
    osdi bsimcmg.osdi
    osdi ekv3.osdi
    osdi juncap200.osdi
    osdi diode_cmc.osdi
    osdi bjt504.osdi
    osdi hicumL2.osdi
    osdi_bind mos 69 psp103va
    osdi_bind mos 69 pspnqs103va has=swnqs
    osdi_bind mos 72 bsimcmg_va  ignore=devtype,version
    osdi_bind mos 79 ekv3_rf     polarity=sign
    osdi_bind bjt 6  bjt504va
    osdi_bind bjt 8  hicumL2va
    osdi_bind dio 6  juncap200   area=ab
    osdi_bind dio 7  diode_cmc

Options:

- `has=<par>`: a level may have several bindings. One with `has=<par>` is
  selected if the card has parameter `<par>` (any value), else the one
  without `has=`. Above, only the NQS module of PSP103 knows `swnqs`.
- `polarity=<par>` (mos, bjt): `nmos`/`npn` sets the module parameter
  `<par>` to 1, `pmos`/`pnp` to -1; default `type`. A contradicting value on the card is
  reported and ignored. `polarity=none`: the module has no polarity.
- `area=<ipar>` (dio): an unlabeled value on a `D` instance
  (`D1 a k dmod 2`) sets instance parameter `<ipar>`; default `area`.
  Named instance parameters are passed to the module unchanged.
- `ignore=<par>,...`: card parameters of other simulators to skip.

Terminals: `M` and `D` instances leave terminals that are not given
unconnected, as `N` instances do. A `Q` instance ties a missing substrate
(4th terminal) to ground, as with built-in models; further terminals, such
as the thermal node, stay internal, so that self-heating acts as the model
card requests (HICUM `flsh`, Mextram `bjt504t`).

On a bound card, `level` selects the binding and is not passed to the
module. A bound level takes precedence over a built-in model of the same
level. The module is looked up when the card is parsed, so `osdi` and
`osdi_bind` may be given in any order. Within a netlist use
`pre_osdi_bind` in a `.control` section. `osdi_bind` without arguments
lists the bindings in the syntax of the command.

## Build Instructions

To compile NGSPICE with OSDI support ensure that the `--enable-predictor` and `--enable-osdi` flags are used.
The `compile_linux.sh` file enables these flags by default.



