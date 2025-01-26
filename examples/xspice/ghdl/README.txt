The circuits in this directory illustrate the use of the d_cosim
XSPICE code model as a container for a VHDL simulation using
the GHDL compiler.  Use a version of GHDL built with the LLVM back-end.

The circuits and steps below are intended to be used from the directory
containing this file, certainly ouput files from GHDL should be in
the current directory when simulating.

The example circuits are:

555.cir: The probably familiar NE555 oscillator provides a minimal example
of combined simulation with SPICE and GHDL.
The digital part of the IC, a simple SR flip-flop, is expressed in VHDL.

pwm.c: VHDL "wait" statements control a pulse-width modulated output to
generate an approximate sine wave.

adc.cir: Slightly more complex VHDL describes the controlling part
of a switched-capacitor ADC.

mc.cir: A motor-control simulation with the control algorithm written
in non-sythesisable VHDL.  It is used as a general-purpose programming
language rather than a HDL.  Apart from showing that this is easy to do,
no part should be taken seriously: the algorithm is poor (but simple),
the motor parameters are invented and the voltages and currents representing
mechanical quantities do not match standard units.

Before a simulation can be run, the associated VHDL code must be compiled
and an additional shared library, ghdlng.vpi must be built.  A library script
isavailable to simplify the steps, run like this:

    ngspice ghnggen adc.vhd

A command interpreter script, ./build or BUILD.CMD, is provided to build
all four examples, with clean/CLEAN.CMD to remove the files created.

