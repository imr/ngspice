#ifndef ngspice_COMPATMODE_H
#define ngspice_COMPATMODE_H

#include "ngspice/config.h"

struct compat
{
	bool hs;
	bool s3;
	bool all;
	bool ps;
	bool lt;
	bool ki;
	bool a;
	bool spe;
};

extern struct compat newcompat;

#endif
