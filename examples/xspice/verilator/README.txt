The circuits in this directory illustrate the use of the d_cosim
XSPICE code model as a container for a Verilog simulation, using the
Verilator compiler. The example circuits are:

555.cir: The probably familiar NE555 oscillator provides a minimal example
of combined simulation with SPICE and Verilog.  The digital part of the IC,
a simple SR flip-flop, is expressed in Verilog.

delay.v: A very simple example of using delays in Verilog to generate
waveform outputs.

pwm.c: Verilog delays controlling a pulse-width modulated output generate
an approximate sine wave.

adc.cir: Slightly more complex Verilog describes the controlling part
of a switched-capacitor ADC.

Before a simulation can be run, the associated Verilog code must be compiled
by Verilator using a command script that is included with ngspice:

    ngspice vlnggen 555.v

That should create a shared library file, 555.so (or 555.DLL on Windows)
that will be loaded by the d_cosim code model.  The compiled Verilog code that
it contains will be executed during simulation.  Similar compilations
are needed to prepare the other examples, but for Verilog with delays the
command looks like:

    ngspice vlnggen -- --timing delay.v

(The "--" prevents "--timing" from being treated as a ngspice option, so it is
passed on to Verilator.)
