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

    osdi_bind mos|bjt|dio <level> <module> [version=<v>] [has=<par>]
                          [polarity=<par>[:<nval>:<pval>]|none]
                          [alias=<name>:<ipar>,...] [ignore=<par>,<par>,...]
    osdi_bind mos|bjt|dio <level> none [version=<v>] [has=<par>]
                                                     remove the binding(s)
    osdi_bind                                        list the bindings

Example for `.spiceinit`:

    osdi psp103.osdi
    osdi psp103_nqs.osdi
    osdi psp104.osdi
    osdi bsimcmg110.osdi
    osdi bsimcmg.osdi
    osdi ekv3.osdi
    osdi juncap200.osdi
    osdi diode_cmc.osdi
    osdi bjt504.osdi
    osdi hicumL2.osdi
    osdi_bind mos 69 psp103va    version=103
    osdi_bind mos 69 pspnqs103va version=103 has=swnqs
    osdi_bind mos 69 psp104va    version=104
    osdi_bind mos 72 bsimcmg     version=110 polarity=devtype:1:0
    osdi_bind mos 72 bsimcmg_va  version=111 ignore=devtype
    osdi_bind mos 79 ekv3_rf     polarity=sign
    osdi_bind bjt 6  bjt504va
    osdi_bind bjt 8  hicumL2va
    osdi_bind dio 6  juncap200   alias=area:ab
    osdi_bind dio 7  diode_cmc

Options:

- `version=<v>` and `has=<par>`: a level may have several bindings. A
  binding applies if the card's `version` starts with `<v>`, not followed
  by a digit (`103` matches `103` and `103.3`, not `1030`), and if the card
  has parameter `<par>` (any value). Of the bindings that apply, the one
  with the most conditions is selected, of these the first given; a
  binding without conditions is the default. Above, only the NQS module of
  PSP103 knows `swnqs`.
- `polarity=<par>` (mos, bjt): `nmos`/`npn` sets the module parameter
  `<par>` to 1, `pmos`/`pnp` to -1; default `type`. Other values with
  `polarity=<par>:<nval>:<pval>`, e.g. `devtype:1:0` for BSIM-CMG 110,
  where `devtype` is 1 for NMOS and 0 for PMOS. A contradicting value on
  the card is reported and ignored. `polarity=none`: the module has no polarity.
- `alias=<name>:<ipar>,...`: on `M`, `Q` and `D` instances of a bound
  card, the instance parameter `<name>` sets the module parameter `<ipar>`.
  The unlabeled value of `D` and `Q` instances (`D1 a k dmod 2`) is the
  parameter `area`, so `alias=area:ab` passes it to `ab`. Other instance
  parameters are passed to the module unchanged.
- `ignore=<par>,...`: card parameters of other simulators to skip, e.g.
  `devtype` of BSIM-CMG 110 cards on the 111 module, which only knows
  `type`.

Terminals: `M` and `D` instances leave terminals that are not given
unconnected, as `N` instances do. A `Q` instance ties a missing substrate
(4th terminal) to ground, as with built-in models; further terminals, such
as the thermal node, stay internal, so that self-heating acts as the model
card requests (HICUM `flsh`, Mextram `bjt504t`).

`R` and `C` instances are not bound: resistor and varactor modules with
more than two terminals (r3_cmc, r2_et_cmc, mosvar) are used with `N`
instances and native model cards (`.model rmod r3_cmc ...`), as in the
IHP PDK.

On a bound card, `level` selects the binding and is not passed to the
module; nor is `version`, unless the module has this parameter. A bound level takes precedence over a built-in model of the same
level. The module is looked up among the OSDI modules when the card is parsed,
so `osdi` and `osdi_bind` may be given in any order, and a module may have
the name of a built-in device (e.g. `bsim4`). Within a netlist use
`pre_osdi_bind` in a `.control` section. `osdi_bind` without arguments
lists the bindings in the syntax of the command.

A binding names a module, not a file. If two loaded `.osdi` files contain
modules of the same name (e.g. two releases of a model compiled from
sources with the same `module` name), only the module loaded first is used;
load only one of them, or give the modules different names.

## Build Instructions

To compile NGSPICE with OSDI support ensure that the `--enable-predictor` and `--enable-osdi` flags are used.
The `compile_linux.sh` file enables these flags by default.



