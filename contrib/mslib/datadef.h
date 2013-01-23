/*
 * MW. Include for spice 
 */

#ifndef ngspice_DATADEF_H
#define ngspice_DATADEF_H

	/*
	 * Program defaults
	 * 
	 * *Directory for input file and input libraries 
	 */
#define DECKPATH "./"
#define LIBPATH "/usr/local/lib/"

	/*
	 * Name for output library file 
	 */
#define TMPLIBNAME ".lib"		/*
								 * * * * actual name is "deck.TMPLIBNAME" 
								 */

	/*
	 * Command for libraries, subckts and models declaration 
	 */
#define LIBINVL "*LIB"
#define SUBINVL "*SUB"
#define MODINVL "*MOD"

	/*
	 * Keywords for subckt start, end and model 
	 */
#define SUBLINE ".SUBCKT"
#define SUBEND ".ENDS"
#define MODLINE ".MODEL"
#define MODEND

#define LIBMESSAGE "*      MW Library include for Spice"

#define BSIZE 255
#define MODDLINE 1
#define SUBDLINE 4
#define LIBDLINE 8
#define SUBLLINE 16
#define MODLLINE 32
#define ENDSLLINE 64
#define CONTLLINE 128
#define NORMLINE 0
#define WRITESUB 0xffff
#define WRITEMOD 0x1111
#define NOWRITE 0x0
#define FAILED 0xffffff
#define SUCCESS 0

#define IS_LIB 0x1
#define LIB_OPEN 0x2

#define IS_MOD 0x10
#define IS_SUB 0x100
#define FINDED 0x400
#define DUPLICATE 0x800

#define DECK_OPEN 0x20000
#define TLIB_OPEN 0x100000

#define NAMEVALID 0xfff
#define NAMENOTV 0x0

struct LSData
  {
	char name[BSIZE];
	FILE *filedes;
	int flag;
	struct LSData *prevLS;
	struct LSData *nextLS;
  };

struct LSData *LSinsert(struct LSData *LS, struct LSData *where);
struct LSData *LSclear(struct LSData *LS);
struct LSData *Backfree(struct LSData *LS);
int readdeck(FILE * tdeck, struct LSData *lib, \
			 struct LSData *sub, struct LSData *mod);
int readlib(struct LSData *lib, FILE * tlib, \
			struct LSData *firstSUB, struct LSData *firstMOD);
int checkname(struct LSData *smp, char *name);

#endif
