README for TclSpice
-------------------

** What is it?

TclSpice is an improved version of Berkeley Spice designed to be used with 
the Tcl/Tk scripting language. The project is open-source (BSD license) 
and based upon the NG-Spice source code base with many improvements

** Features and Improvements

    * Fully Tcl scriptable - installs with 'package require spice' statement
    * Hspice syntax (SpicePP).
    * GUI interfaces, various (Tk).
        - SpiceWish (BLT graph widget)
    * BLT (tcl compatible) vectors for storage, manipulation and arithmetic 
        upon Spice waveforms.
    * Xspice additions (Georgia Tech).
    * Trigger upon waveform events.
    * Spice 'simulator state' save and restore for rapid 'what-if' simulations 
       (no longer need to re-simulate from the beginning each time a 
       device value is changed).

** INSTALL

    * Requirements: (over normal ngspice)

        Tcl/Tk 8.3 or higher compiled with thread support
        BLT-2.4u for Tcl or newer ( http://www.sourceforge.net/projects/blt/ )
        tclreadline ( http://tclreadline.sourceforge.net/ )

        IMPORTANT:
           If Tcl/Tk is not compiled with thread support ( --enable-threads 
           configure argument ) then tclspice will be quite unstable.

    * Building and installing

        $ tar -zxf tclspice0.2.??.tar.gz
        $ cd tclspice
        $ ./configure --with-tcl --enable-experimental --enable-xspice
        $ make tcl
        $ make install-tcl

        IMPORTANT:
           Use make tcl and make install-tcl otherwise tclspice will fail
           to compile.

** USAGE

    $ spicewish <spicefile>


** More information:

    src/tcl/README : more in depth information on how to create your
       own Tcl scripts. 
       Example also supplied in the same directory (vector_test.tcl).

    src/xspice/README : More details on the xspice additions / useage

    http://tclspice.sourceforge.net/ : Homepage of Tclspice

    Stefan Jones <stefan.jones@multigig.com>
    ngspice-devel@lists.sourceforge.net : Email support!
