#ifndef ngspice_ENHTYPES_H
#define ngspice_ENHTYPES_H


typedef enum {
    ENH_ANALOG_NODE,            /* An analog node */
    ENH_EVENT_NODE,             /* An event-driven node */
    ENH_ANALOG_BRANCH,          /* A branch current */
    ENH_ANALOG_INSTANCE,        /* An analog instance */
    ENH_EVENT_INSTANCE,         /* An event-driven instance */
    ENH_HYBRID_INSTANCE,        /* A hybrid (analog/event-driven) instance */
} Enh_Conv_Source_t;


typedef struct Enh_Bkpt Enh_Bkpt_t;
typedef struct Enh_Ramp Enh_Ramp_t;
typedef struct Enh_Conv_Debug Enh_Conv_Debug_t;
typedef struct Enh_Conv_Limit Enh_Conv_Limit_t;
typedef struct Enh_Rshunt Enh_Rshunt_t;
typedef struct Enh_Ckt_Data Enh_Ckt_Data_t;


#endif
