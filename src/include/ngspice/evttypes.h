#ifndef ngspice_EVTTYPES_H
#define ngspice_EVTTYPES_H
#include "miftypes.h"

typedef struct Evt_Output_Info Evt_Output_Info_t;
typedef struct Evt_Port_Info Evt_Port_Info_t;
typedef struct Evt_Inst_Index Evt_Inst_Index_t;
typedef struct Evt_Node_Info Evt_Node_Info_t;
typedef struct Evt_Inst_Info Evt_Inst_Info_t;
typedef struct Evt_Info Evt_Info_t;
typedef struct Evt_Inst_Event Evt_Inst_Event_t;
typedef struct Evt_Inst_Queue Evt_Inst_Queue_t;
typedef struct Evt_Node_Queue Evt_Node_Queue_t;
typedef struct Evt_Output_Event Evt_Output_Event_t;
typedef struct Evt_Output_Queue Evt_Output_Queue_t;
typedef struct Evt_Queue Evt_Queue_t;
typedef struct Evt_Node Evt_Node_t;
typedef struct Evt_Node_Data Evt_Node_Data_t;
typedef struct Evt_State Evt_State_t;
typedef struct Evt_State_Desc Evt_State_Desc_t;
typedef struct Evt_State_Data Evt_State_Data_t;
typedef struct Evt_Msg Evt_Msg_t;
typedef struct Evt_Msg_Data Evt_Msg_Data_t;
typedef struct Evt_Statistic Evt_Statistic_t;
typedef struct Evt_Data Evt_Data_t;
typedef struct Evt_Count Evt_Count_t;
typedef struct Evt_Limit Evt_Limit_t;
typedef struct Evt_Job Evt_Job_t;
typedef struct Evt_Option Evt_Option_t;
typedef struct Evt_Ckt_Data Evt_Ckt_Data_t;
typedef struct Evt_Node_Cb Evt_Node_Cb_t;

typedef Mif_Boolean_t (*Evt_New_Value_Cb_t)(double when, Mif_Value_t *val_p,
                                            void *ctx, int is_last);

typedef enum Evt_Node_Cb_Type { Evt_Cbt_Raw, Evt_Cbt_Plot} Evt_Node_Cb_Type_t;

#endif
