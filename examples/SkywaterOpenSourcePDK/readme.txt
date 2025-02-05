* simple examples using Skywater PDK

PDK sources:
https://github.com/google/skywater-pdk
http://opencircuitdesign.com/open_pdks/

Examples are: a simple inverter, and a benchmark circuit ISCAS85 C7552 with ca. 15k transistors.
These examples serve as low level starters with given ngspice netlist. If you are to design a circuit,
you will be better off with a GUI atop ngspice like XSCHEM, see 
http://repo.hu/projects/xschem/xschem_man/tutorial_run_simulation.html and many other web sites and videos.

How to run the circuits in ngspice:

Edit the input files for correcting the path to the Skywater library, as it is user and OS specific.

Create a file .spiceinit in your HOME directory (or in this project directory). 
It should contain the following statements:

set num_threads=8 ; number of physical core in your machine
set ngbehavior=hsa ; set compatibility for reading PDK libs
set skywaterpdk ; skip some checks for faster lib loading 
set ng_nomodcheck ; trust the models, don't check the model parameters          
unset interactive ; run scripts without interruption (default)
set nginfo ; some more info
option KLU ; select KLU solver over Sparse 1.3 solver

Run the simulation by calling
ngspice skywater_inverter.net

A plotting alternative in addition to the internal graphics is gnuplot.
