/* Modified: 2000 AlansFixes */
#include "ngspice/ngspice.h"
#include "dup2.h"

#ifndef HAVE_DUP2
#include <fcntl.h>

dup2(int oldd, int newd)
{
#ifdef HAVE_FCNTL_H
	close(newd);
#ifdef F_DUPFD
	(void) fcntl(oldd, F_DUPFD, newd);
#endif
#endif
	return 0;
}
#else
int Dummy_Symbol_6;
#endif
