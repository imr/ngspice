*********************************************************************************
*Copied and written by Stefan Jones (stefan.jones@multigig.com) at Multigig Ltd *
*Code based on and copied from ScriptEDA                                        *
*(http://embedded.eecs.berkeley.edu/Alumni/pinhong/scriptEDA/)                  *
*Copyright (C) 2001   Author  Pinhong Chen                                      *
*                                                                               *
*This program is free software; you can redistribute it and/or                  *
*modify it under the terms of the GNU Lesser General Public License             *
*as published by the Free Software Foundation; either version 2                 *
*of the License, or (at your option) any later version.                         *
*                                                                               *
*This program is distributed in the hope that it will be useful,                *
*but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
*MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
*GNU Lesser General Public License for more details.                            *
*                                                                               *
*You should have received a copy of the GNU Lesser General Public License       *
*along with this program; if not, write to the Free Software                    *
*Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.*
*********************************************************************************

WARNING!

The text in this document has been prepared in 2006 and is partially outdated.
It is provided here only for reference and may provide some (historical)
information.


Welcome to tclspice README_tcl

This file acompanies ngspice sources starting with ngspice-rework-18. It
describes what tclspice is, its installation, and points to resources that will
help you to start with it. It also contains usefull datas to keep informed,
get help, or get involved in the project.

Enjoy

Lionel (saintel@users.sourceforge.net)



What is tclspice:

tclspice is a variant of ngspice. It share 95% of its code (approx). The
different between plain NGspice and tclspice is the type of binary it produces,
and the way to access it. NGspice is a standalone program that you can execute
and which can either automatically process a spice directives script. It can
also propose you a command line interface.
tclspice is a tcl package name. It is based on libspice shared library. This
library is specifically designed to be loaded in tcl interpreters such as tclsh
or wish. Then all spice directives are available as tcl functions. libspice also
feature some new commands that are usefull for the integration into the tcl
environment.

tclspice differs from ngspice by its printf (bundled to tcl printf), malloc (tcl
 malloc), data handling and plotting (BLT toolkit extensions to tcl/tk).


Installing:

Tclspice relies on three packages that are not included in ngspice:
tcl : the tcl command interpreter, for interpretion of user scipt
tk  : the graphical extension of tcl, to represent data graphically and for GUIs
blt : BLT toolkit gives number handling and plotting features to tcl/tk

The latest configuration is (not much tested):
tclspice-27
tcl 8.4
tk 8.4
blt 2.4

Tclspice is built and installed the same way as ngspice. Then, after reading this
paragraph, the information you lack will probably be in README file in this directory.
There is a configuration flag to set in order to compile tclspice library rather than
plain ngspice, that is a standalone program. This flag is --with-tcl. It accepts an
argument the path to tclConfig.sh

If you don't provide any argument, configure script will try to find it automatically.
  ./configure --enable-xspice --disable-cider --disable-xgraph --enable-numparam --with-readline=no --enable-adms=no --with-tcl

If its does not, then it will propose you some possible locations.
  can't find Tcl configuration script "tclConfig.sh"
  Should you add --with-tcl=/usr/lib/tcl8.4/tclConfig.sh to ./configure arguments?

If it does not, check that tcl8.4 is installed, and manually specify the path.
  ./configure --enable-xspice --disable-cider --disable-xgraph --enable-numparam --with-readline=no --enable-adms=non --with-tcl=/usr/lib/tcl8.4

Tclspice is not compatible with ngspice graphical code. Then when building tclspice,
--no-x is automatically configured.



Support and help :

First of all (but last recourse), mail me at saintel@users.sourceforge.net

For any kind of information on tclspice:
  http://ngspice.sourceforge.net/tclspice.html
It gives plenty of information. There is an index of good resources that you can
read to get into tclspice quickly and proficently.
