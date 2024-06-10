This directory contains Ngspice scripts and other files used to prepare
Verilog code to be included in an Ngspice simulation, using Verilator
or Icarus Verilog.


For Verilator the relevant files are vlnggen (an Ngspice script),
verilator_main.cpp and verilator_shim.cpp.  The two C++ files are
compiled together with C++ source code generated from the Verilog input.
The compilation is handled by Verilator unless the Microsoft Visual C++
compiler is used.  MSVC.CMD contains an example command for that.

Example circuits can be found in examples/xspice/verilator.


The following files are for Icarus Verilog support and are built into
shared libraries while compiling Ngspice: icarus_shim.h, icarus_shim.c,
vpi.c, vpi_dummy.c, user_vpi_dummy.h, coroutine*.h and libvvp.def.

Example circuits can be found in examples/xspice/icarus_verilog.
