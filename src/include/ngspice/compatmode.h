#ifndef ngspice_COMPATMODE_H
#define ngspice_COMPATMODE_H

#include "ngspice/config.h"

struct compat
{
	bool hs; /* HSPICE */
	bool s3; /* spice3 */
	bool ll; /* all */
	bool ps; /* PSPICE */
	bool lt; /* LTSPICE */
	bool ki; /* KiCad */
	bool a; /* whole netlist */
	bool spe; /* spectre */
	bool eg; /* EAGLE */
};

extern struct compat newcompat;

#endif
