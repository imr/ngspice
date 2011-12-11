#include "ngspice/cpdefs.h"
#include <string.h>
#include "hcomp.h"

int
hcomp(const void *a, const void *b)
{
    struct comm **c1 = (struct comm **) a;
    struct comm **c2 = (struct comm **) b;
    return (strcmp((*c1)->co_comname, (*c2)->co_comname));
}
