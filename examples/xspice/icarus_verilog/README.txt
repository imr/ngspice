The circuits in this directory illustrate the use of the d_cosim
XSPICE code model as a container for a Verilog simulation using
Icarus Verilog.  Icarus Verilog must be built with the --enable-libvvp option,
so that its simulation engine is available as a dynamic library.
The Verilog source code and included parts of the circuit definitions
can be found in the adjacent "verilator" directory.

The example circuits are:

555.cir: The probably familiar NE555 oscillator provides a minimal example
of combined simulation with SPICE and Verilog.
The digital part of the IC, a simple SR flip-flop, is expressed in Verilog.

delay.v: A very simple example of using delays in Verilog to generate
waveform outputs.

pwm.c: Verilog delays controlling a pulse-width modulated output generate
an approximate sine wave.

adc.cir: Slightly more complex Verilog describes the controlling part
of a switched-capacitor ADC.

Before a simulation can be run, the associated Verilog code must be compiled:

    iverilog -o 555 ../verilator/555.v

Similar compilations are needed to prepare the other examples.

The simulations require additional dynamic libraries, ivlng.so (or ivlng.DLL)
and ivlng.vpi: they are expected to be in the usual installation location.

To use the versions in a built source tree that has not been installed,
the .model definitions in the circuit files must be changed to the ugly:

.model vlog_ff d_cosim sim_args=["555"]
+ simulation = "../../../release/src/xspice/verilog/.libs/ivlng"
+ lib_args = [ "libvvp"
+ "../../../release/src/xspice/verilog/.libs/ivlngvpi.so" ]

Or for Windows builds using MSVC:

.model vlog_ff d_cosim sim_args=["555"]
+ simulation = "..\..\..\visualc\xspice\verilog\ivlng.DLL"
+ lib_args = [ "libvvp"
+ "..\..\..\visualc\xspice\verilog\shim.vpi" ]

