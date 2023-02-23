#ifndef ngspice_EVT_H
#define ngspice_EVT_H

/* ===========================================================================
FILE    EVT.h

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of the evt data structure and all
    its substructures.  The single evt structure is housed inside of
    the main 3C1 circuit structure 'ckt' and contains virtually all
    information about the event-driven simulation.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"
#include "ngspice/miftypes.h"
#include "ngspice/evttypes.h"



/* ************** */
/* Info structure */
/* ************** */


struct Evt_Output_Info {
    Evt_Output_Info_t  *next;         /* the next in the linked list */
    int  node_index;                  /* index into node info struct for this output */
    int  output_subindex;             /* index into output data in node data struct  */
    int  inst_index;                  /* Index of instance the port is on */
    int  port_index;                  /* Index of port the output corresponds to */
};

struct Evt_Port_Info {
    Evt_Port_Info_t         *next;        /* the next in the linked list of node info */
    int                     inst_index;   /* Index of instance the port is on */
    int                     node_index;   /* index of node the port is connected to */
    char                    *node_name;   /* name of node port is connected to */
    char                    *inst_name;   /* instance name */
    char                    *conn_name;   /* connection name on instance */
    int                     port_num;     /* port number of instance connector */
};

struct Evt_Inst_Index {
    Evt_Inst_Index_t         *next;     /* the next in the linked list */
    int                      index;     /* the value of the index */
};

struct Evt_Node_Info {
    Evt_Node_Info_t   *next;           /* the next in the linked list */
    char              *name;           /* Name of node in deck */
    int               udn_index;       /* Index of the node type */
    Mif_Boolean_t     invert;          /* True if need to make inverted copy */
    Mif_Boolean_t     save;            /* Save data for this node */
    int               num_ports;       /* Number of ports connected to this node */
    int               num_outputs;     /* Number of outputs connected to this node */
    int               num_insts;       /* The number of insts receiving node as input */
    Evt_Inst_Index_t  *inst_list;      /* Linked list of indexes of these instances */
};

struct Evt_Inst_Info {
    Evt_Inst_Info_t         *next;      /* the next in the linked list of node info */
    MIFinstance             *inst_ptr;  /* Pointer to MIFinstance struct for this instance */
};

struct Evt_Info {
    Evt_Inst_Info_t    *inst_list;         /* static info about event/hybrid instances */
    Evt_Node_Info_t    *node_list;         /* static info about event nodes */
    Evt_Port_Info_t    *port_list;         /* static info about event ports */
    Evt_Output_Info_t  *output_list;       /* static info about event outputs */
    int                *hybrid_index;      /* vector of inst indexs for hybrids */
    Evt_Inst_Info_t    **inst_table;       /* vector of pointers to elements in inst_list */
    Evt_Node_Info_t    **node_table;       /* vector of pointers to elements in node_list */
    Evt_Port_Info_t    **port_table;       /* vector of pointers to elements in port_list */
    Evt_Output_Info_t  **output_table;     /* vector of pointers to elements in output_list */
};








/* *************** */
/* Queue structure */
/* *************** */



struct Evt_Inst_Event {
    Evt_Inst_Event_t  *next;        /* the next in the linked list */
    double            event_time;   /* Time for this event to happen */
    double            posted_time;  /* Time at which event was entered in queue */
};

struct Evt_Inst_Queue {
    Evt_Inst_Event_t  **head;          /* Beginning of linked lists */
    Evt_Inst_Event_t  ***current;      /* Beginning of pending events */
    Evt_Inst_Event_t  ***last_step;    /* Values of 'current' at last accepted timepoint */
    Evt_Inst_Event_t  **free;          /* Linked lists of items freed by backups */
    double            last_time;       /* Time at which last_step was set */
    double            next_time;       /* Earliest next event time in queue */
    int               num_modified;    /* Number modified since last accepted timepoint */
    int               *modified_index; /* Indexes of modified instances */
    Mif_Boolean_t     *modified;       /* Flags used to prevent multiple entries */
    int               num_pending;     /* Count of number of pending events in lists */
    int               *pending_index;  /* Indexes of pending events */
    Mif_Boolean_t     *pending;        /* Flags used to prevent multiple entries */
    int               num_to_call;     /* Count of number of instances that need to be called */
    int               *to_call_index;  /* Indexes of instances to be called */
    Mif_Boolean_t     *to_call;        /* Flags used to prevent multiple entries */
};




struct Evt_Node_Queue {
    int               num_to_eval;    /* Count of number of nodes that need to be evaluated */
    int               *to_eval_index; /* Indexes of nodes to be evaluated */
    Mif_Boolean_t     *to_eval;       /* Flags used to prevent multiple entries */
    int               num_changed;    /* Count of number of nodes that changed */
    int               *changed_index; /* Indexes of nodes that changed */
    Mif_Boolean_t     *changed;       /* Flags used to prevent multiple entries */
};




struct Evt_Output_Event {
    Evt_Output_Event_t    *next;      /* the next in the linked list */
    double            event_time;     /* Time for this event to happen */
    double            posted_time;    /* Time at which event was entered in queue */
    Mif_Boolean_t     removed;        /* True if event has been deactivated */
    double            removed_time;   /* Time at which event was deactivated */
    void              *value;         /* The delayed value sent to this output */
};

struct Evt_Output_Queue {
    Evt_Output_Event_t  **head;          /* Beginning of linked lists */
    Evt_Output_Event_t  ***current;      /* Beginning of pending events */
    Evt_Output_Event_t  ***last_step;    /* Values of 'current' at last accepted timepoint */
    Evt_Output_Event_t  ***free_list;    /* Pointers to linked lists of items freed by backups */
    double              last_time;       /* Time at which last_step was set */
    double              next_time;       /* Earliest next event time in queue */
    int                 num_modified;    /* Number modified since last accepted timepoint */
    int                 *modified_index; /* Indexes of modified outputs */
    Mif_Boolean_t       *modified;       /* Flags used to prevent multiple entries */
    int                 num_pending;     /* Count of number of pending events in lists */
    int                 *pending_index;  /* Indexes of pending events */
    Mif_Boolean_t       *pending;        /* Flags used to prevent multiple entries */
    int                 num_changed;     /* Count of number of outputs that changed */
    int                 *changed_index;  /* Indexes of outputs that changed */
    Mif_Boolean_t       *changed;        /* Flags used to prevent multiple entries */
};




struct Evt_Queue {
    Evt_Inst_Queue_t   inst;               /* dynamic queue for instances */
    Evt_Node_Queue_t   node;               /* dynamic queue of changing nodes */
    Evt_Output_Queue_t output;             /* dynamic queue of delayed outputs */
};




/* ************** */
/* Data structure */
/* ************** */




struct Evt_Node {
    Evt_Node_t       *next;           /* pointer to next in linked list */
    Mif_Boolean_t    op;              /* true if computed from op analysis */
    double           step;            /* DC step or time at which data was computed */
    void             **output_value;  /* Array of outputs posted to this node */
    void             *node_value;     /* Resultant computed from output values */
    void             *inverted_value; /* Inverted copy of node_value */
};

struct Evt_Node_Data {
    Evt_Node_t     **head;          /* Beginning of linked lists */
    Evt_Node_t     ***tail;         /* Location of last item added to list */
    Evt_Node_t     ***last_step;    /* 'tail' at last accepted timepoint */
    Evt_Node_t     **free;          /* Linked lists of items freed by backups */
    int            num_modified;    /* Number modified since last accepted timepoint */
    int            *modified_index; /* Indexes of modified nodes */
    Mif_Boolean_t  *modified;       /* Flags used to prevent multiple entries */
    Evt_Node_t     *rhs;            /* Location where model outputs are placed */
    Evt_Node_t     *rhsold;         /* Location where model inputs are retrieved */
    double         *total_load;     /* Location where total load inputs are retrieved */
};




struct Evt_State {
    Evt_State_t          *next;        /* Pointer to next state */
    Evt_State_t          *prev;        /* Pointer to previous state */
    double               step;         /* Time at which state was assigned (0 for DC) */
    void                 *block;       /* Block of memory holding all states on inst */
};


struct Evt_State_Desc {
    Evt_State_Desc_t        *next;   /* Pointer to next description */
    int                     tag;     /* Tag for this state */
    int                     size;    /* Size of this state */
    int                     offset;  /* Offset of this state into the state block */
};


struct Evt_State_Data {
    Evt_State_t    **head;              /* Beginning of linked lists */
    Evt_State_t    ***tail;             /* Location of last item added to list */
    Evt_State_t    ***last_step;        /* 'tail' at last accepted timepoint */
    Evt_State_t    **free;              /* Linked lists of items freed by backups */
    int            num_modified;        /* Number modified since last accepted timepoint */
    int            *modified_index;     /* List of indexes modified */
    Mif_Boolean_t  *modified;           /* Flags used to prevent multiple entries */
    int            *total_size;         /* Total bytes for all states allocated */
    Evt_State_Desc_t **desc;            /* Lists of description structures */
};




struct Evt_Msg {
    Evt_Msg_t            *next;      /* Pointer to next state */
    Mif_Boolean_t        op;         /* true if output from op analysis */
    double               step;       /* DC step or time at which message was output */
    char                 *text;      /* The value of the message text */
    int                  port_index; /* The index of the port from which the message came */
};


struct Evt_Msg_Data {
    Evt_Msg_t      **head;              /* Beginning of linked lists */
    Evt_Msg_t      ***tail;             /* Location of last item added to list */
    Evt_Msg_t      ***last_step;        /* 'tail' at last accepted timepoint */
    Evt_Msg_t      **free;              /* Linked lists of items freed by backups */
    int            num_modified;        /* Number modified since last accepted timepoint */
    int            *modified_index;     /* List of indexes modified */
    Mif_Boolean_t  *modified;           /* Flags used to prevent multiple entries */
};


struct Evt_Statistic {
    int                op_alternations;    /* Total alternations between event and analog */
    int                op_load_calls;      /* Total load calls in DCOP analysis */
    int                op_event_passes;    /* Total passes through event iteration loop */
    int                tran_load_calls;    /* Total inst calls in transient analysis */
    int                tran_time_backups;  /* Number of transient timestep cuts */
};




struct Evt_Data {
    Evt_Node_Data_t    *node;               /* dynamic event solution vector */
    Evt_State_Data_t   *state;              /* dynamic event instance state data */
    Evt_Msg_Data_t     *msg;                /* dynamic event message data */
    Evt_Statistic_t    *statistics;         /* Statistics for events, etc. */
};



/* **************** */
/* Counts structure */
/* **************** */


struct Evt_Count {
    int          num_insts;             /* number of event/hybrid instances parsed */
    int          num_hybrids;           /* number of hybrids parsed */
    int          num_hybrid_outputs;    /* number of outputs on all hybrids parsed */
    int          num_nodes;             /* number of event nodes parsed */
    int          num_ports;             /* number of event ports parsed */
    int          num_outputs;           /* number of event outputs parsed */
};



/* **************** */
/* Limits structure */
/* **************** */


struct Evt_Limit {
    int         max_event_passes;    /* maximum loops in attempting convergence of event nodes */
    int         max_op_alternations; /* maximum loops through event/analog alternation */
};


/* ************** */
/* Jobs structure */
/* ************** */


struct Evt_Job {
    int                num_jobs;           /* Number of jobs run */
    int                cur_job;            /* job selected for plotting etc */
    char               **job_name;         /* Names of different jobs */
    char               **job_plot;         /* Names of different plots created by the job */
    Evt_Node_Data_t    **node_data;        /* node_data for different jobs */
    Evt_State_Data_t   **state_data;       /* state_data for different jobs */
    Evt_Msg_Data_t     **msg_data;         /* messages for different jobs */
    Evt_Statistic_t    **statistics;       /* Statistics for different jobs */
};



/* ***************** */
/* Options structure */
/* ***************** */


struct Evt_Option {
    Mif_Boolean_t   op_alternate;        /* Alternate analog/event solutions in OP analysis */
};


/* ****************** */
/* Main evt structure */
/* ****************** */

struct Evt_Ckt_Data {
    Evt_Count_t     counts;         /* Number of insts, nodes, etc. */
    Evt_Info_t      info;           /* Static info about insts, etc. */
    Evt_Queue_t     queue;          /* Dynamic queued events */
    Evt_Data_t      data;           /* Results and state data */
    Evt_Limit_t     limits;         /* Iteration limits, etc. */
    Evt_Job_t       jobs;           /* Data held from multiple job runs */
    Evt_Option_t    options;        /* Data input on .options cards */
};



#endif
