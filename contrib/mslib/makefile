#	MW. Include libs for Spice
CFLAGS=		-O2 -s
LDFLAGS=	-s

OBJS=	inc_main.o inc_inp.o inc_LSD.o
HDRS=	datadef.h
SRCC=	inc_main.c inc_inp.c inc_LSD.c

mslib:	$(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) 

clean:
	$(RM) $(OBJS) mslib

$(OBJS):	$(HDRS)
