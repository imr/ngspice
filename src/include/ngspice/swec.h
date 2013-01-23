/*
 *  project.h 
 *
 *  Timing Simulator (ESWEC)
 *
 *  Date: October 5, 1990
 *
 *  Author: Shen Lin
 *
 *  Copyright (C) University of California, Berkeley
 *
 */
#ifndef ngspice_SWEC_H
#define ngspice_SWEC_H

/************************************************************
 *
 *	Defines
 *
 ************************************************************/

#define MainTitle "     Timing  Simulator\n"
#define MAXDEVICE       4
#define MAXMOS        31500         /* suggested value  */
#define MAXDD          256         /* suggested value  */
#define MAXVCCS        128         /* suggested value  */
#define MAXTIME 1000000
#define MAXNODE       136

#define TAB_SIZE            8192  /* originally 2048 */
#define NUM_STEPS_PER_MICRON  10  /* 0.1 micron is the smallest step */
#define MAX_FET_SIZE          80  /* largest fet in microns */
#define Vol_Step          1.0e-3  /* voltage resolution */
#define SCL               1000.0  /* voltage scaler (1V/3mv) */
#define MAX_CP_TX_LINES        8  /* max number of coupled lines in
				     a multiconductor line system  */

/************************************************************
 *
 *	Macro
 *
 ************************************************************/
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#endif

/************************************************************
 *
 *	Data Structure Definitions
 *
 ************************************************************/

typedef struct reglist           REGLIST;
typedef struct node              NODE;
typedef struct mosfet            MOSFET;
typedef struct emosfet           EMOSFET;
typedef struct diode             DIODE;
typedef struct ediode            EDIODE;
typedef struct vccs              VCCS;
typedef struct evccs             EVCCS;
typedef struct i_cap             I_CAP;
typedef struct ei_cap            EI_CAP;
typedef struct resistor          RESISTOR;
typedef struct eresistor         ERESISTOR;
typedef struct rline             RLINE;
typedef struct erline            ERLINE;
typedef struct txline            TXLine; 
typedef struct etxline           ETXLine; 
typedef struct cpline            CPLine;
typedef struct ecpline           ECPLine;
typedef struct bqueue            BQUEUE;
typedef struct pqueue            PQUEUE; 
typedef struct ms_device         MS_DEVICE;
typedef struct bp_device         BP_DEVICE;
typedef struct dd_device         DD_DEVICE;

struct mosfet{
   int          type;     /* 1 : NMOS, 2 : PMOS */
   MS_DEVICE    *device;  /* NULL if the nominal device model */
   NODE         *out_node; 
   NODE         *in_node; 
   float        Cs, Cd;
   MOSFET       *nx;
   int          time;          /*  instantaneous information  */
   float        voltage, dvg;  /*  instantaneous information  */
   float        vgN_1;      /*  gate voltage at previous event point  */
   float        G;          /*  effective conductance at t(n)  */
   float        effective;  /*  W over effective L  */
   int          tabW;       /*  width in ns/um  */
   REGLIST      *region;  /*  region associated with this mos  */
			  /*  NULL if driven by the node of the same region  */
};

struct diode{
   float        Is;       /* saturation current */
   float        Vj;       /* junction potential */
   double       G;
   NODE         *in_node;
   NODE         *out_node;
   DIODE        *nx;
};

struct vccs{
   float        Is;       /* saturation current */
   NODE         *in_node;
   NODE         *out_node;
   NODE         *pcv_node;
   NODE         *ncv_node;
   DIODE        *nx;
};

typedef struct {
   char         name[10];        /* device name */
   int          type;            /* 1 : NMOS, 2 : PMOS */
   int          device_id;       /* device id */
} DEVICENAME;
   
struct ms_device{
   char name[10];
   int  used;          /* device used in circuit flag */
   float rho;          /* device vsat denom param */
   float alpha;        /* device vsat denom vgg param */
   float vt;           /* device zero bias threshold voltage in mv*/
   float gamma;        /* device backgate bias vt param */
   float fermi;        /* device fermi potential in mv */
   float theta;        /* device backgate bias vt width param */
   float mu;           /* device vt width param */
   float eta;          /* device saturation slope */
   float eta5;         /* eta - 0.5 */
   int   pzld;         /* positive lambda */
   float lambda;       /* channel-length modulation */
   float kp;           /* device conductance parameter */
   float cgs0;         /* gate-source overlap capacitance 
			    per meter channel width */
   float cgd0;         /* gate-drain overlap capacitance
			    per meter channel width */
   float cox;          /* oxide-field capacitance 
			    per square meter of gate area */
   float cjsw;         /* zero-biased junction sidewall capacitace 
			    per meter of junction perimeter */
   float cj0;          /* zero-biased junction bottom capacitace
                            per square meter of junction area */
   float keq;          /* abrupt junction parameter */

   float ld;           /* lateral diffusion */
   float *thresh;
   float *sat;
   float *dsat;
   float *body;
   float *gammod;
};

struct bp_device{
   char name[10];
   int   type;         /* 1 : NPN; 2 : PNP */
   float rc;           /* collector resistance */
   float re;           /* emitter resistance */
   float rb;           /* zero bias base resistance */
   float Is;           /* transport saturation current */
   float Af;           /* ideal maximum forward alpha */
   float Ar;           /* ideal maximum reverse alpha */
   float Vje;          /* B-E built-in potential */
   float Vjc;          /* B-C built-in potential */
};

struct dd_device{
   char name[10];
   float Is;           /* saturation current */
   float rs;           /* ohmic resistance */
   float Vj;           /* junction potential */
};

typedef struct linked_lists_of_Bpoint{
   struct linked_lists_of_Bpoint *next;
   int time;
   float voltage;
   float slope;  
} BPOINT, *BPOINTPTR;

typedef struct linked_lists_of_nodeName{
   char      id[24]; 
   struct    linked_lists_of_nodeName  *left, *right;
   NODE      *nd;
}  NDname, *NDnamePt;
  
struct node {
   NDnamePt  name; 
   EMOSFET   *mptr;    /* pointer to head of src/drn MOSFET list */
   EMOSFET   *gptr;    /* pointer to head of gate MOSFET list */
   EI_CAP    *cptr;    /* pointer to head of internodal cap list */
   ERESISTOR *rptr;    /* pointer to head of internodal resistor list */
   ERLINE    *rlptr;   /* pointer to head of internodal TX line  list */
   ETXLine   *tptr;    /* pointer to head of transmission line list */
   ECPLine   *cplptr;  /* pointer to head of coupled lines list */
   EDIODE    *ddptr;   /* pointer to head of diode list */
   EVCCS     *vccsptr; /* pointer to head of VCCS list */
   EVCCS     *cvccsptr;/* pointer to head of controlled VCCS list */
   NODE      *next;    /* pointer to next node */
   REGLIST   *region;   /* region associated with this node */
   NODE      *base_ptr; /* group src/drn nodes into region */
   /* charles 2,2 1/18/93 
   float     V;           
   float     dv;       voltage at t(n-1) and slope at t(n)  
   */
   double	 V;
   double    dv;
   double    CL;       /*  grounded capacitance in F  */
   double    gsum;     /*^ sum of the equivalent conductance */
   double    cgsum;    /*^ sum of the constant conductance */
   double    is;       /*^ equivalent Is */
   int       tag;      /*  -2 : Vdd, -3 : Vss, -1 : initial value  */
   int       flag;     /*^ flag to show some features of the node */
   PQUEUE    *qptr;    /*^ pointer to the entry in the queue or waiting list */
   FILE      *ofile;   /*  output file for the signal at this node  */
		       /*  NULL if not for print  */
   int dvtag;
};

struct reglist{
   REGLIST     *rnxt;   /*  pointer to next region  */
   NODE        *nlist;  /*  node list  */
   MOSFET      *mos;
   I_CAP       *cap;
   RESISTOR    *res;
   TXLine      *txl;
   CPLine      *cpl;
   struct linked_lists_of_Bpoint *Bpoint; /* break points at primary inputs */
   struct linked_lists_of_Bpoint *head; /* header of the break points at primary inputs */
   int eTime; /*  time when this region previously evaluated  */
   int DCvalue;
     /*  1, 0, 2 : unknown, 3 : unchangeable 1, 4 : unchangeable 0  */
   BQUEUE      *prediction;
};


struct bqueue{
   int    key;     /* time for the event to be fired, or DC weight */
   BQUEUE   *left;
   BQUEUE   *right;
   BQUEUE   *pred;
   BQUEUE   *pool;
   REGLIST  *region;  /* region id */
};

struct  pqueue { 
   NODE     *node;
   PQUEUE   *next;
   PQUEUE   *prev;
};

struct i_cap {
   NODE         *in_node;
   NODE         *out_node;
   float        cap;
   I_CAP        *nx;
};

struct resistor {
   NODE         *in_node;
   NODE         *out_node;
   float        g;   /* conductance */ 
   int          ifF; /* whether floating */
   float        g1;  /* conductance for floating resistor */
   RESISTOR     *nx;
};

struct rline {
   NODE         *in_node;
   NODE         *out_node;
   double        g;   /* conductance */ 
   RLINE        *nx;
};

typedef struct linked_lists_of_vi_txl{
   struct linked_lists_of_vi_txl *next;
   struct linked_lists_of_vi_txl *pool;
   int time;
   /* charles 2,2 
   float v_i, v_o;
   float i_i, i_o;  
   */
   double v_i, v_o;
   double i_i, i_o;  
} VI_list_txl;

typedef struct linked_lists_of_vi{
   struct linked_lists_of_vi *next;
   struct linked_lists_of_vi *pool;
   int time;
   double v_i[MAX_CP_TX_LINES], v_o[MAX_CP_TX_LINES];
   double i_i[MAX_CP_TX_LINES], i_o[MAX_CP_TX_LINES];  
} VI_list;

typedef struct {
   double c, x;
   double cnv_i, cnv_o;
} TERM;

typedef struct {
   int    ifImg;
   double aten;
   TERM   tm[3];
} TMS;

struct cpline {
   int       noL;
   int       ext;
   double     ratio[MAX_CP_TX_LINES];
   double     taul[MAX_CP_TX_LINES];
   TMS       *h1t[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   TMS       *h2t[MAX_CP_TX_LINES][MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   TMS       *h3t[MAX_CP_TX_LINES][MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double    h1C[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double    h2C[MAX_CP_TX_LINES][MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double    h3C[MAX_CP_TX_LINES][MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double    h1e[MAX_CP_TX_LINES][MAX_CP_TX_LINES][3];
   NODE      *in_node[MAX_CP_TX_LINES];
   NODE      *out_node[MAX_CP_TX_LINES];
   int       tag_i[MAX_CP_TX_LINES], tag_o[MAX_CP_TX_LINES];
   CPLine    *nx;
   struct linked_lists_of_vi *vi_head;
   struct linked_lists_of_vi *vi_tail;
   double     dc1[MAX_CP_TX_LINES], dc2[MAX_CP_TX_LINES];
};

struct txline {
   int       lsl;  /*  1 if the line is lossless, otherwise 0  */
   int       ext;  /*  a flag, set if time step is greater than tau  */
   double    ratio;
   double    taul;
   double    sqtCdL;
   double    h2_aten;
   double    h3_aten;
   double    h1C;
   double    h1e[3];
   int       ifImg;
   NODE      *in_node;
   NODE      *out_node;
   int       tag_i, tag_o;
   TERM      h1_term[3];
   TERM      h2_term[3];
   TERM      h3_term[6];
   TXLine    *nx; 
   struct linked_lists_of_vi_txl *vi_head;
   struct linked_lists_of_vi_txl *vi_tail;
   double    dc1, dc2;
   int	     newtp; /* flag indicating new time point */
};

struct evccs {
   VCCS       *vccs;
   EVCCS      *link;
};

struct ediode {
   DIODE       *dd;
   EDIODE      *link;
};

struct emosfet {
   MOSFET       *mos;
   EMOSFET      *link;
};

struct ei_cap {
   I_CAP        *cap;
   EI_CAP       *link;
};

struct eresistor {
   RESISTOR     *res;
   ERESISTOR    *link;
};

struct erline {
   RLINE        *rl;
   ERLINE    *link;
};

struct etxline {
   TXLine    *line;
   ETXLine   *link; 
};

struct ecpline {
   CPLine    *line;
   ECPLine   *link;
};

#endif
