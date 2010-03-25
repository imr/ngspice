********************************************************
README.TXT         the minimal Numparam documentation
********************************************************

Numparam: an add-on library for electronic circuit analysis front-ends
Copyright (C)  2002    Georg Post

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


CONTENTS

A.  The Numparam library 
B.  Circuit description with Numparam (in lieu of a tutorial)
C.  Reference section
D.  Installation
E.  Theory of operation
F.  Files in this package
G.  Known bugs


A.  The Numparam library:

The spice-3f(x) front end lacks some features of commercial derivatives,
such as the ability to define numerical attributes of devices by symbols
or by constant (at circuit-expansion time) numerical expressions.
Numerical parameters - other than node names - for subcircuits are also
commonly available in these non-free Spices.

This library provides a retrofit to the Spice circuit description language
to add these features. By design, the new code is organized as an independent
library which does not import any Spice header files and  whose only interface
to Spice is a set of three function headers. The only place where these
functions are called - if a new compile-time option for Spice is set - is
the code file src/lib/fte/subckt.c.  There are no patches anywhere else.

The purpose of this minimal coupling was to freely license the additional code.
It is distributed under the GNU-LGPL and thus can be linked to the original
Spice which is open-source software but maintains a special license.
(As I read somewhere, Berkeley Spice is reserved to people friendly to the USA).
Due to GPL-type requirements, I cannot merge any lines from Spice with
the new code, and I cannot avoid redundancies, such as one more symbol table
manager and the umpteenth arithmetic expression parser. 

Coding style of my add-on is a bit personal. Using a set of keyword-hiding
macros, it is closer to Pascal and/or Basic than to authentic C programs.
Most of it originated from my initial Turbo Pascal preprocessors that have been
translated to C. After all, I'm definitely not "friendly to the C language".
Even in disguise, the code is pure Ansi-C and compiles without a warning
under the severest settings of gcc (under GNU/Linux) or Turbo C++ (under
MS-DOS). For C hardliners, I include the 'washprog' utility which downgrades
the source code to the common C look-&-feel. Extreme purists might apply
some appropriate "indent" utility, in addition. 


B. Circuit description with Numparam (in lieu of a tutorial).

As we now know, all the text entry to Spice comes in two separate languages:
- a circuit description language (CDL) which defines the electronic hardware.
- an analysis job and data management language (JDML) which may be used
   as an interactive shell or in batch files.

In the circuit description "CDL" file, the design is typically organized
as a hierarchical set of subcircuits which are connected together in
the 'main' circuit (they are "called", like procedures in a general-purpose
programming language). CDL is line-oriented, with a recognizer role assigned
to the first non-space character of a line.
(In the ancient times of data processing, a line was called a "card".)
For example,  '*' introduces comment lines. 'C' places a capacitor
device in the circuit, 'R' a resistor, 'L' an inductor, and so on.
'X' signals the "call" of a subcircuit. Character '+' introduces a continuation
line (the way to split exceedingly long lines into pieces).

A special class of lines that start with a dot '.' have control functions
inside CDL: they do not add physical circuit elements. 
For instance, the control pair '.subckt' and '.ends' brackets a subcircuit
definition section in CDL.
In the old days of Spice-2, some of the functions now assigned to JDML were
also inserted as dot cards. You can still insert pieces of JDML inside
a CDL file, as a section enclosed within lines '.control'  and  '.endc' . 

Example CDL file, a netlist of some double T  RC-filter:

* First-example
.subckt myfilter in out
Ra in p1   2k
Rb p1 out  2k
C1 p1 0    2nF
Ca in p2   1nF
Cb p2 out  1nF
R1 p2 0    1k
.ends myfilter

X1 input output myfilter
V1 input 0 AC 1V
.end

Let us recall what the Spice "front-end" essentially does to your 
circuit-description (CDL) file whenever it is submitted, either at program
start-up or after some interactive JDML commands like 'edit' or 'source'.
First, all the JDML sections in the file are sorted out and kept for
later use (unless the file is pure JDML and thus immediately executed). 
Next, the hierarchy of subcircuits is expanded and an internal representation
of the flattened circuit is stored, i.e. a set of CDL lines without any more
'X' and '.subckt' lines. This flat circuit is also known as the "netlist".
Then, the netlist is translated into the internal data structures of Spice,
essentially a sparse matrix of elements indexed by pairs of circuit nodes.

Finally, the mathematical analysis is carried out under the control of JDML,
and output data may be stored, printed, plotted, compared, and so on. 
Analyses may be repeated under varying bias/frequency/time... conditions.
But to change the circuit topology, the CDL must be edited and re-compiled.

Numparam-specific features of CDL :

The Numparam library is an enhancement of the Spice front-end which adds
clarity and arithmetic functionality to the circuit description language.

The most wanted feature of a language is to have word-like symbols that
take the place of specific values or objects. The dot-line

.param <identifier> = <expression>

defines such symbols. For example, to describe a triple RC filter
with identical values of components, we do not need to explicitly
repeat constant numbers. The CDL code may go like this:

   .param res= 1kohm // numparam allows comment tails  like in C++
   .param tau= 10ns  // we want a time constant
   .param cap= tau/res  // tau= RC, of course

   .subckt triplerc in out
   R1 in  p1  {res}
   C1 p1  0   {cap}
   R2 p1  p2  {res}
   C2 p2  0   {cap}
   R3 p2  out {res}
   C3 out 0   {cap}
   .ends 

As you can see, the use of symbols anywhere in the circuit description
requires the curly braces :
  { <expression> }
This coding style is even more interesting if circuit elements have known
fixed ratios (Butterworth filters and the like) and we only need to
touch one value (a time constant) to tune the circuit.

Only numerical constants such as '30pF' may be used without enclosing
braces. It is the braces that tell our CDL front-end to look up
symbols and to crunch arithmetic expressions inside.

Obviously, it was restrictive that subcircuit definitions could only
accept interface node names as symbolic parameters. With the following
syntax of the .subckt line, we add numerical parameters:

  .subckt <ckt-ident> <node> ... params: <id>=<value> <id>=<value> ...

Example, a parameterized filter:

  .subckt triplerc2 in out params: res=1kohm cap=50pF
   *  all the following lines as above.
   *  the specified default values are always overridden with X lines.

To call variants of such a subcircuit later on, we may write:

  X1 in out1 triplerc2 {r1}    {c1}
  X2 in out2 triplerc2 {2*r1}  {c1/2}
  X3 in out3 triplerc2 {3*r1}  {c1/3}

where the r1 and c1 symbols are defined in .param lines.
So, we can use subcircuits with one or more parameters, the same way 
as Spice2 already allowed an area parameter for diodes and transistors.


Here is the first example, rewritten with parameters:

* Param-example
.param amplitude= 1V

.subckt myfilter in out
+ params: rval=100k  cval= 100nF
Ra in p1   {2*rval}
Rb p1 out  {2*rval}
C1 p1 0    {2*cval}
Ca in p2   {cval}
Cb p2 out  {cval}
R1 p2 0    {rval}
.ends myfilter

X1 input output myfilter 1k 1nF
V1 input 0 AC {amplitude}
.end

Note:
Now, there is some possible confusion in Spice because of multiple numerical
expression features. The .param lines and the braces expressions are
evaluated in the front-end, that is, just after the subcircuit expansion.
(Technically, the X lines are kept as comments in the expanded circuit
so that the actual parameters can correctly be substituted ).
So, after the netlist expansion and before the internal data setup, all
number attributes in the circuit are known constants.
   However, there are some circuit elements in Spice which accept arithmetic
expressions that are NOT evaluated at this point, but only later during
circuit analysis. These are the arbitrary current and voltage sources.
The syntactic difference is that "compile-time" expressions are
within braces, but "run-time" expressions have no braces.
  To make things more complicated, the backend language JDML also accepts
arithmetic/logic expressions that operate on its own scalar or vector data sets.

It would be desirable to have the same expression syntax, operator and function
set, and precedence rules, for the three contexts mentioned above.
In the current Numparam implementation, that goal is not yet achieved...


C.  Reference section:

The Numparam add-on supports the following elements in the circuit description
language.

1. '.param'  control lines to define symbolic numbers
2. arithmetic expressions in place of any numeric constant
3. formal and actual numeric parameters for subcircuit definition and 'call'. 

NOT YET IMPLEMENTED:
  To activate the additional functions, put a line near the top of the CDL file:
  .option numparam 

In the syntax description,
<ident> means an alphanumeric identifier (<20 chars, starting with a letter)
<expr>  means an expression, composed of <ident>s, Spice numbers, and operators.

1. The .param line:
   Syntax:   .param <ident> = <expr> ; <ident> = <expr> ....

  This line assigns numerical values to identifiers. More than one assignment
  per line is possible using the ';' separator. 
  The .param lines inside subcircuits are copied per call, like any other line.
  All assignments are executed sequentially through the expanded circuit.
  Before its first use, a name must have been assigned a value.
  
2. Brace expressions in cicuit elements:
   Syntax:     { <expr> }

  These are allowed in .model lines and in device lines, wherever only constant
  Spice numbers could be used in spice2/3. A Spice number is a floating
  point number with an optional scaling suffix, immediately glued to the
  numeric tokens (see below). 
  Warning: {..} cannot be used to 'parameterize' node names or parts of names.
    ( We are not into  obfuscated shell scripting ...)
  All identifiers used within an <expr> must have known values at the time 
  when the line is evaluated, else an error is flagged.

3. Subcircuit parameters:
   The syntax of a subcircuit definition header is:
     .subckt <ident> node node ... params: <ident>= <value> <ident>=<value>...

   node is an integer number or an identifier, for one of the external nodes.
   The 'params:' keyword introduces an optional section of the line.
   Each <ident> is a formal parameter, and each <value> is either a Spice
   number or a brace expression.
   Inside the '.subckt' ... '.ends' context, each formal parameter may be
   used like any identifier that was defined on a .param control line.
   The <value> parts are supposed to be default values of the parameters.
   However, in the current version of Numparam, they are not used and each
   invocation of the subcircuit must supply the _exact_ number of actual
   parameters.

   The syntax of a subcircuit call (invocation) is:
       X<name>  node node ... <ident>  <value> <value> ....
    
   Here <name> is the symbolic name given to that instance of the subcircuit,
   <ident> is the name of a subcircuit defined beforehand.  node node ... is  
   the list of actual nodes where the subcircuit is connected. 
   <value> is either a Spice number or a brace expression { <expr> } .
   The sequence of <value> items on the X line must exactly match the number
   and the order of formal parameters of the subcircuit.

4. Symbol scope

   All Subcircuit and Model names are considered global and must be unique.
   The .param symbols that are defined outside of any '.subckt' ... '.ends'
   section are global. Inside such a section, the pertaining 'params:' 
   symbols and any .param assignments are considered local: they mask any
   global identical names, until the .ends line is encountered.
   You cannot reassign to a global number inside a .subckt, a local copy is
   created instead. Scope nesting now works up to any level. For example,
   if the main circuit calls A which has a formal parameter xx, A calls B
   which has a param. xx, and B calls C which also has a formal param. xx,
   there will be three versions of 'xx' in the symbol table but only the most
   local one - belonging to C - is visible.  

5. Syntax of expressions <expr>  ( optional parts within [ ...] ):

    An expression may be one of:
       <atom>       where <atom> is either a Spice number or an identifier 
       <unary-operator> <atom>
       <function-name> ( <expr> [ , <expr> ...] )
       <atom> <binary-operator> <expr>         
       ( <expr> )

    As expected, atoms, builtin function calls and stuff within parentheses
    are evaluated before the other operators. The operators are evaluated
    following a list of precedence close to the one of the C language. 
    For equal precedence binary ops, evaluation goes left to right.

    Operators:   Alias     Internal symb.      Precedence

     -                       -                   1     (unary -)
     not            !        !                   1     (unary not)
     **             ^        ^                   2     (power)
     *                       *                   3     (multiply)
     /                       /                   3     (divide)
     mod            %        %                   3     (modulo)
     div            \        \                   3     (integer divide)
     +                       +                   4     (add)
     -                       -                   4     (subtract)
     ==                      =                   5     (equality)
     <>             !=       #                   5     (un-equal)
     <=                      L                   5     (less or equal)
     >=                      G                   5     (greater or equal)
     <                       <                   5     (less than)
     >                       >                   5     (greater than)
     and            &&       &                   6     (and)
     or             ||       |                   7     (or)   
     
    The result of logical operators is 1 or 0 , for True or False.


    Builtin functions:      Internal ref.

    defined                 0     (returns 1 if symbol is defined, else 0)
    sqr                     1
    sqrt                    2
    sin                     3
    cos                     4
    exp                     5
    ln                      6
    arctan                  7
    abs                     8
    pwr                     9

    Scaling suffixes (any decorative alphanum. string may follow ...)

    g    1e9
    meg  1e6
    k    1e3
    m    1e-3
    u    1e-6
    n    1e-9
    p    1e-12
    f    1e-15 

  Note: there are intentional redundancies in expression syntax, e.g. 
    x^y ,  x**y    and  pwr(x,y)  all have nearly the same result.     
  
6.  Reserved words
   In addition to the above function names and to the verbose operators   
   ( not and or div mod ), other words are reserved and cannot be used
   as parameter names. Historically, they come from a version of Numparam
   that was a full-blown macro language. I won't link that one to Spice,
   not before somebody proves to me that such a thing could be useful...   
  
     and or not div mod if else end while macro funct defined
     include for to downto is var 
     sqr sqrt sin cos exp ln arctan abs pwr 


7. Alternative syntax
   the & sign is tolerated to provide some 'historical' parameter notation:
   & as the first character of a line is equivalent to:  .param
   Inside a line, the notation &(....) is equivalent to {....}, and
   &identifier means the same thing as {identifier} .

   This notation exists a bit for the same reason as my macros which wipe
   the curly braces out of the C language: entering those signs is a pain in
   the neck on IBM French-type keyboards. You hit, among others, a vanishingly
   small AltGr key which is squeezed by superfluous buttons that show ugly 
   office-software logos...

   Comments in the style of C++ line trailers (//) are detected and erased.
   Warning: this is NOT possible in embedded .control parts of a source
   file, these JDML lines are outside of Numparam's scope. DOS-style
   carriage returns at line ends are difficult for JDML, too.


D.  Installation

There are two versions of Spice on which this library has been tried:
a.  a 1997 version spice3f5 that was arranged for Red Hat Linux
b.  the  version 14 of ngspice (will now be privileged for development)

On my system, the size of libnupa.a is about 47k, so this is the additional
bloat that the spice3 and nutmeg  binary programs will accumulate.
( The numparam source tarball weighs in for some 70k )

The common part to build the Numparam library is this:

0. choose any directory you like for Numparam,  let's call it $HACK.
1. un-tar  the .c and .h files and the rest, in Numparam's directory :
    tar xzvf numparam.tgz    
  
2. compile the lib sources  with gcc -c -Wall:

  gcc -c -ansi -pedantic -Wall spicenum.c nupatest.c xpressn.c mystring.c

3. pre-link together the library part to numparam.o and libnupa.a:

  ld -r -o numparam.o spicenum.o xpressn.o mystring.o
  ar -rcs libnupa.a   spicenum.o xpressn.o mystring.o

4. make the test executable nupatest:

    gcc -o nupatest nupatest.o spicenum.o xpressn.o mystring.o -lm

The script file 'mknumpar.sh' does all this (2-4).


5a.  Link with the "third version of Spice3f5 for RedHat Linux 2.6" (1997)

 do the following in the spice3f5 top-level directory: 

 1. patch the file src/lib/fte/subckt.c :
      cp -biv $HACK/rhsubckt.c  src/lib/fte/subckt.c  
 2.  edit  src/bin/makeops , to add $HACK/libnupa.a  to LIBS and LIBN.
 3. ./util/build linux
 4. ./util/build linux install


5b.  Link procedure for ngspice version 14

I haven't yet a working knowledge of the 'automake/autoconf' system, so I'll
describe the pedestrian hacks to get Numparam in. That's evil; the right way
would need a configuration flag that chooses to make and to link the library. 
Only the top level files 'configure.in' and 'Makefile.am' should be revised
to process the numparam option. (?)
Help!

1. replace  the file src/frontend/subckt.c with Numparam's patched version :
      cp -biv  $HACK/ngsubckt.c  src/frontend/subckt.c  
2. run ./configure with a "LIBS prefix" to include  numparam (see below)
3. make
4. make install

Here is one "prefixed" ngspice configure script that works on my system:
 
#!/bin/sh

#  ngconfig.sh
#     configure options for ngspice with numparam add-on
#     run this in  ngspice's top-level directory 

# specify your Numparam directory
HACK=/home/post/spice3f5/hack

# over-write the original subckt.c
cp -biv $HACK/ngsubckt.c  src/frontend/subckt.c  

# my box needs CFLAGS on 1st run, else 'terminal.c' wont find 'termcap.h' ?

CFLAGS=-I/usr/include/ncurses \
LIBS=$HACK/libnupa.a \
./configure --without-x --prefix=/usr/local/ngsp

####  end of sample script  ####


E.  Theory of operation

Spice's front end does a lot of malloc/free type memory gymnastics and does not
seem to care much about small leaks here and there. Numparam will do some
malloc'ing in place of Spice (essentially the translated strings of the input
deck) and rely on Spice to clean it up - or not - later on. My library will  
clean up its private space only (the symbol tables) and will make some 
assumptions about the interface function calls coming from Spice.
Here is the scenario supposed to be followed by Spice and Numparam:

0. the patched codefile subckt.c imports the following header lines:

#define  NUPADECKCOPY 0
#define  NUPASUBSTART 1
#define  NUPASUBDONE  2
#define  NUPAEVALDONE 3

extern char * nupa_copy(char *s, int linenum);
extern int    nupa_eval(char *s, int linenum);
extern int    nupa_signal(int sig); 

These are the three library functions called, i.e.

-  nupa_copy by inp_subcktexpand to preprocess all extended-syntax lines.
-  nupa_eval by inp_subcktexpand to do the parameter substitutions
-  nupa_signal with one of the 4 signals, from various places to
                send state information to the Numparam library.

The only places with numparam patches  are the functions
inp_subcktexpand() and its recursive subroutine doit(), in the
file subckt.c . At this stage, we suppose that:
- any .control sections are filtered out
- any .include are expanded
- any + continuation line chunks are glued together 

1. In the first phase, Numparam runs through the deck (whose .control sections
   have already been removed by Spice) to create copies of the lines
   without the extended syntax. Pointers to the original deck lines are kept
   and the copies are traditional Spice, with placeholders for
   symbols and expressions. Spice loses the originals and gets the bleached-out
   copies.

2. The "doit()"  circuit expansions are modified to keep more information.
   Contrary to the initial Spice code, now the subcircuit invocation
   lines are preserved as comments, which allows Numparam to update
   symbolic subcircuit parameters a bit later. Subcircuit exit lines are also
   copied and out-commented, to keep track of identifier scopes during
   the final pass.

If this seems waste of storage, just consider all those sloppy memory
leaks in the circuit expansion code...

3. The final wash-up is a sequential call to the library (nupa_eval())
   line-by-line through the expanded circuit. By using its pointers
   to the original lines, Numparam recovers the added syntax features.
   It triggers all the symbol value computations and inserts constant
   numbers into the circuit definition lines, whose length must not change!
   This option is a kludge to avoid memory reallocation [ my intuitive
   fear is that these free() malloc() realloc() and friends  swallow a lot of
   CPU time ? ].

4. The termination signal at the end of inp_subcktexpand() tells the Numparam
   library to clean up its mess, release its pointers to the original
   Spice circuit description text, and to get prepared for another run. 
   Note: Numparam frees the storage related to the original lines
   whose pointers have been stolen in phase 1.


In a future release, Numparam will be re-entrant, all its 'global' data being
referenced via a handle which the client program should keep around.


F.  Files in this package

The following Ansi C code files belong to Numparam:

general.h    header file with macros to disguise the C language. 
             stuff for an 'overflow-safe' string library ( whose biggest bug
             is that it indexes strings from 1 like Pascal).

numparam.h   header file for numparam-specific symbols and functions 

mystring.c   collection of 'safer' character string (and misc.) functions.
             beware of the nasty Turbo Pascal conventions.

xpressn.c    the interpreter of arithmetic/logical expressions

spicenum.c   the interface part, functions that are called by Spice.

nupatest.c   a stand-alone subcircuit expander, for test purpose. 

washprog.c   a program that washes all the above C files, including itself,
             to recover the crude syntax of the True Language (see below).

Patched versions of spice's  subckt.c file incorporate the library calls
and maybe try to repair some memory leaks (in rhsubckt.c, not yet tested).

rhsubckt.c    for spice3f5 1997 Red Hat  (src/lib/fte/subckt.c)
ngsubckt.c    for ngspice version 14     (src/frontend/subckt.c) 
subckt.dif    'diff' between ngsubckt.c and ngspice frontend/subckt.c

The following text, data and script files  are also included:

readme.txt     this documentation file
downgrad.txt   the substitution rules required for washprog.c
mknumpar.sh    script to make the library binaries
ngconfig.sh    sample script to run ./configure for ngspice

configure.in   crappy ?
Makefile.am    crappy ?

testfile.nup   a test text (nonsense circuit) for Numparam ?


So, if you are a Real Programmer, think that the Pascal amateurs confound
programming with writing novels, and find those Basic greenhorns' style
too childish,  then execute the following two-liner first of all
(should work on GNU/Linux, but it's not a speed monster) :

gcc -o washprog washprog.c mystring.c
./washprog  *.c

You get all the *.c files in a version where the first character becomes an
underbar, and the interior resembles to _code_. (although it lacks such
powerful features as  continue, break, goto, ?:-expressions, gets(), ... )


G. Known Bugs

First of all, lots of size limits  -  incompatible with the Spirit of the
Gnu, who wants that everything may grow as much as malloc() can grab ...  

- circuit source code line length:      80 chars
- circuit '+' extended code lines:      250 chars 
- number of source lines:               1000
- number of lines in expanded circuit:  5000
- length of numparam identifiers:       20 chars
- number of numparam identifiers:       200
- length of file names:                 80 chars
- significant digits in param results:  5
- nesting depth of parentheses          9
- nesting of subckt calls               10

All these constants should be in the header file but aren't.

After each circuit expansion, numparam asks a silly question
of the "Abort/Continue" type. A debugging feature, to be killed soon.

The Numparam symbol table accumulates the following sets of names:
subcircuits, models, global parameters, subcircuit arguments.
Node names, however, are completely ignored.

Call the following "bugs" or "features":
-  A model/subckt name cannot be defined twice, even if local to a subcircuit.
-  The same name cannot design a model here, and a parameter elsewhere.
-  A subcircuit argument masks any global parameter of same name,
   anytime the subckt is invoked. Inside a .subckt context, .param assignments 
   also have local scope and override global identical names.
 
It is wise to always use unique names for everything.


While Numparam is in 'early beta stage', I strongly suggest to use
'nupatest' first, on any 'parameterized' Spice circuit file,
before starting the enhanced circuit analyser.

The command  
  nupatest foobar.cir
produces an output file 'foobar.out' which is the expanded and
parameter-reduced flat netlist.
By the way, it produces error messages whenever it chokes on the source file.
If nupatest succeeds, the spice+numparam combo should swallow it, too.
Big bug: Nupatest does not yet prefix and infix things inside v() and i().


Numparam comes with two very experimental files 'configure.in' and
'Makefile.am' as an exercise of the automake/autoconf mechanisms.
I certainly got a lot of things wrong and had to do _eight_ steps to
have it kind of work:

1. edit/create configure.in
2. edit/create Makefile.am
3. run         autoheader                                 --> config.h.in
4. run         automake --foreign --add-missing --verbose --> Makefile.in
5. run         aclocal                                    --> aclocal.m4
6. run         autoconf                                   --> configure
7. run         ./configure                                --> Makefile  config.h
8. run         make

Do we need all this, and -worse- do we need to repeat it whenever we touch
'configure.in' and/or 'Makefile.am'  ?  Help!


Please send your bug reports, improvements, flames etc. to the author:
georg.post @ wanadoo.fr

