/**********
Enhancement-242: native n-port rational-model device.

A built-in ngspice device that realizes an arbitrary-port linear block from a
pole-residue (vector-fitted) Y-parameter model produced by `pre_snp -native`:

    Y_ij(s) = d_ij + s * e_ij + sum_k  res_ijk / (s - p_k)          (shared poles)

Stamped DIRECTLY into the sparse matrix (DC/AC/tran) in admittance / branch-current
form -- no Verilog-A / OpenVAF compile -- so it scales to hundreds of ports where
the `pre_snp -osdi` (VA->OSDI) path hits the compiler wall (~24-32 ports).

Instantiated through the generic `N` device dispatcher (inp2n.c), broadened from
OSDI-only to also accept this model type:

    N1  p1 p2 ... pN  ref   mymodel
    .model mymodel nport(file="mymodel.nport")

Port nodes are read from the generic GENnode() array (ports 0..N-1, then ref).
The fit data lives in a compact `.nport` file, written by `pre_snp -native`, and
is read into the model at temperature time.
**********/

#ifndef ngspice_NPORTDEFS_H
#define ngspice_NPORTDEFS_H

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/klu.h"        /* BindElement (KLU CSC binding) */

/* Max terminals accepted through the N dispatcher for an nport instance
 * (ports + 1 reference).  Sizing the generic GENnode array; instances that use
 * fewer ports simply leave the rest unbound. */
#define NPORT_MAXTERMS 512

/* per-instance data */
typedef struct sNPORTinstance {

    struct GENinstance gen;

    /* GENnode array -- CKTbindNode() writes the port + reference node numbers
     * here, and GENnode(inst) returns (int*)(inst+1), so this MUST be the very
     * first member after `gen` (before any other field), sized to the device's
     * terminal count (NPORT_MAXTERMS).  ports are [0..N-1], reference is [N]. */
    int  NPORTnodeArray[NPORT_MAXTERMS];

#define NPORTmodPtr(inst)       ((struct sNPORTmodel *)((inst)->gen.GENmodPtr))
#define NPORTnextInstance(inst) ((struct sNPORTinstance *)((inst)->gen.GENnextInstance))
#define NPORTname               gen.GENname
#define NPORTstate              gen.GENstate

    int  NPORTn;               /* number of ports on this instance (== model N) */
    int  NPORTrefNode;         /* reference node index (GENnode[N])             */

    /* Direct admittance (multi-terminal conductance) stamp -- no branch
     * currents.  Port current leaving node i is I_i = sum_j Y_ij (V_j - V_ref),
     * giving the four-corner stamp for each (i,j):
     *   (node_i, node_j) += +Y_ij     (ref, node_j) += -Y_ij
     *   (node_i, ref)    += -Y_ij     (ref, ref)    += +Y_ij   (accumulated)
     * Complex AC uses the (ptr, ptr+1) real/imag convention (RLC/OSDI style). */
    double **NPORTyPtr;        /* [N*N] (node_i, node_j)           */
    double **NPORTyColPtr;     /* [N]   (node_i, ref) : -sum_j Y_ij */
    double **NPORTyRowPtr;     /* [N]   (ref, node_j) : -sum_i Y_ij */
    double  *NPORTyRefPtr;     /* (ref, ref) : +sum_ij Y_ij         */

    /* KLU CSC bindings, parallel to the pointer arrays above (NULL for a
     * ground row/col, which keeps its Sparse trash pointer, exactly like the
     * built-in RLC devices). */
    BindElement **NPORTyBind;      /* [N*N] */
    BindElement **NPORTyColBind;   /* [N]   */
    BindElement **NPORTyRowBind;   /* [N]   */
    BindElement  *NPORTyRefBind;   /* single */

    /* transient companion state base (Phase 3) */
    int  NPORTstateBase;

    unsigned NPORTallocated :1;   /* setup arrays allocated */
} NPORTinstance;


/* per-model data */
typedef struct sNPORTmodel {

    struct GENmodel gen;

#define NPORTmodType         gen.GENmodType
#define NPORTnextModel(inst) ((struct sNPORTmodel *)((inst)->gen.GENnextModel))
#define NPORTinstances(inst) ((NPORTinstance *)((inst)->gen.GENinstances))
#define NPORTmodName         gen.GENmodName

    char  *NPORTfile;          /* path to the .nport fit file */

    /* fit data, loaded from NPORTfile */
    int     NPORTnPorts;       /* N  */
    int     NPORTnPoles;       /* Np (real-canonical layout: real poles first,
                                *     conj pairs as adjacent +Im then its mate) */
    double *NPORTpoleRe;       /* [Np] */
    double *NPORTpoleIm;       /* [Np] */
    double *NPORTd;            /* [N*N] constant term d_ij */
    double *NPORTe;            /* [N*N] s-linear  term e_ij */
    double *NPORTresRe;        /* [N*N*Np] residue real res_ijk (index (i*N+j)*Np+k) */
    double *NPORTresIm;        /* [N*N*Np] residue imag res_ijk */

    unsigned NPORTfileGiven :1;
    unsigned NPORTloaded    :1;
} NPORTmodel;

/* model parameters */
enum {
    NPORT_MOD_NPORT = 1,   /* nport() -- the .model type flag (bare keyword) */
    NPORT_MOD_FILE,        /* file="..." */
    NPORT_NPORTS,          /* query: N  */
    NPORT_NPOLES           /* query: Np */
};

/* load the fit file into the model (nportread.c) */
extern int NPORTreadFile(NPORTmodel *model);

#include "nportext.h"

#endif /* ngspice_NPORTDEFS_H */
