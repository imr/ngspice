#include <config.h>
#include <ngspice.h>

#include <jobdefs.h>
#include <cktdefs.h>

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
};


char *spice_analysis_get_name(int index)
{
    return analInfo[index]->public.name;
}

char *spice_analysis_get_description(int index)
{
    return analInfo[index]->public.description;
}

int spice_num_analysis(void)
{
    return sizeof(analInfo)/sizeof(SPICEanalysis*);
}


SPICEanalysis **spice_analysis_ptr(void)
{
    return (SPICEanalysis **) analInfo;
}
