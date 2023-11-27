The circuit adc.cir in this directory illustrates the use of the d_cosim
XSPICE code model as a container for a Verilog simulation.  Before the
simulation can be run, the Verilog code must be compiled by Verilator
using the command:

    ngspice vlnggen adc.v

That should create a shared library file, adc.so (or adc.DLL on Windows)
that will be loaded by the d_cosim code model.  The compiled Verilog code that
it contains will be executed during simulation.
