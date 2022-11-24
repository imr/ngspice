#ifndef ngspice_COMPATMODE_H
#define ngspice_COMPATMODE_H

#include "ngspice/config.h"

struct compat
{
	bool isset; /* at least one mode is set */
	bool hs; /* HSPICE */
	bool s3; /* spice3 */
	bool ll; /* all */
	bool ps; /* PSPICE */
	bool lt; /* LTSPICE */
	bool ki; /* KiCad */
	bool a; /* whole netlist */
	bool spe; /* spectre */
	bool eg; /* EAGLE */
	bool mc; /* to be set for 'make check' */
	bool xs; /* XSPICE */
};

extern struct compat newcompat;

#endif
