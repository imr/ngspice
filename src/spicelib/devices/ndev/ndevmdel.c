/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NDEVmDelete(GENmodel *gen_model)
{
    NDEVmodel *model = (NDEVmodel *) gen_model;

    close(model->sock);
    printf("Disconnect to remote NDEV server %s:%d\n", model->host, model->port);

    return OK;
}
