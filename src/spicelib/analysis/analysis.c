#include <config.h>
#include "ngspice/ngspice.h"

#include "ngspice/jobdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

extern SPICEanalysis *analInfo[];
extern SPICEanalysis OPTinfo;
extern SPICEanalysis ACinfo;
extern SPICEanalysis DCTinfo;
extern SPICEanalysis DCOinfo;
extern SPICEanalysis TRANinfo;
extern SPICEanalysis PZinfo;
extern SPICEanalysis TFinfo;
extern SPICEanalysis DISTOinfo;
extern SPICEanalysis NOISEinfo;
extern SPICEanalysis SENSinfo;
/* SP: Periodic Steady State*/
extern SPICEanalysis PSSinfo;
/* SP */

SPICEanalysis *analInfo[] = {
    &OPTinfo,
    &ACinfo,
    &DCTinfo,
    &DCOinfo,
    &TRANinfo,
    &PZinfo,
    &TFinfo,
    &DISTOinfo,
    &NOISEinfo,
    &SENSinfo,
    &PSSinfo,
};


char *spice_analysis_get_name(int index)
{
    return analInfo[index]->if_analysis.name;
}

char *spice_analysis_get_description(int index)
{
    return analInfo[index]->if_analysis.description;
}

int spice_num_analysis(void)
{
    return NUMELEMS(analInfo);
}


SPICEanalysis **spice_analysis_ptr(void)
{
    return (SPICEanalysis **) analInfo;
}
