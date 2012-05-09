#
# Makefile for Code Model directories
#

# Include global XSPICE selections for CC and other macros
include /usr/local/xspice-1-0/include/make.include

INCLUDE = -I. -I$(ROOT)/include/sim

CFLAGS = -g

#-----------------------------------------------------------------------------
# Edit the following definition to specify the object files that comprise
# your code model. If your code model is completely specified in the
# cfunc.mod file, there is no need to edit this definition.
# DO NOT include the ifspec.o file.

CODE_MODEL_OBJECTS = cfunc.o

#-----------------------------------------------------------------------------
# DO NOT MODIFY THE FOLLOWING DEFINITIONS:

.SUFFIXES: $(SUFFIXES) .mod .ifs

.mod.c:
	$(BINDIR)/cmpp -mod $<

.ifs.c:
	$(BINDIR)/cmpp -ifs

.c.o: $*.c
	${CC} ${CFLAGS} ${INCLUDE} -c $*.c

all : ifspec.o $(CODE_MODEL_OBJECTS)

cfunc.o : cfunc.c
ifspec.o : ifspec.c
