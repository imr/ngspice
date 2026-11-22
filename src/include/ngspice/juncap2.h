/**********
 * JUNCAP2 200.6 junction model (current and charge), shared code
 *
 * Used by device models that offer JUNCAP2 for their junctions, e.g. the
 * BSIM4 source/drain junctions (BSIM4 model parameter JUNCAP = 2).  The
 * host model embeds a JC2model in its model struct, a JC2tdata (temperature
 * dependent values) in its model struct or, if the instances can have
 * different temperatures, in its instance struct, and a JC2inst per
 * junction in its instance struct.  It puts the parameter table entries
 * into its tables with an ID base of its own, and calls
 *
 *   JC2mParam / JC2mAsk   model parameter set / ask (IDs relative to the base)
 *   JC2setup              defaults, from the setup function
 *   JC2temp               temperature update, from the temperature function
 *   JC2initInst           geometry of one junction of an instance
 *   JC2eval               current, conductance, charge and capacitance
 *
 * Parameter names follow JUNCAP2-in-PSP 103.  A two-terminal device uses
 * the source set (plain names: CJORBOT, ...).  A MOSFET has a second set
 * for the drain with suffix D (CJORBOTD, ...); as in PSP 103 it is used
 * only with SWJUNASYM = 1, otherwise the drain uses the source parameters.
 * With SWJUNASYM = 1 a drain parameter that is not given takes the value
 * of the corresponding source parameter.
 *
 * Reference temperature: TRJ if given, else a default of the host model
 * (argument trdef of JC2temp and JC2mAsk), chosen as in other simulators:
 * BSIM4 (juncap=2) TNOM if given in the model card, else 27 C; diode level 6
 * 21 C as in the Verilog-A code, TNOM and TREF have no effect there.  The
 * option TNOM never changes it.  TR and TREF are aliases of TRJ where the
 * host model has no parameter of that name.
 *
 * IFACTOR and CFACTOR are instance parameters, as in the Verilog-A code.
 *
 * Differences found in comparisons with a reference simulator, which
 * otherwise follows the Verilog-A code: it uses eps_Si = 1.045e-10 F/m
 * instead of EPSO * 11.8, and in JUNCAP-express the reverse limit
 * xhighr = ln(IMAX/|ISATREV|) instead of ln(0.5*IMAX/|ISATREV|).  It ignores
 * the drain parameters, SWJUNASYM, IFACTOR, CFACTOR and FREV (FREV is used
 * with its default 1e3).
 **********/

#ifndef ngspice_JUNCAP2_H
#define ngspice_JUNCAP2_H

#include "ngspice/ifsim.h"

/* JUNCAP2 component index */
#define JC2_BOT 0
#define JC2_STI 1
#define JC2_GAT 2

#define JC2_NOLIMIT 1.0e300

/*
 * Per-junction parameter list (source names; drain = name + "d").
 * X(a, name, default, lower bound, upper bound, description), a is passed
 * through.  Bounds as in JUNCAP200_parlist.include; values outside are
 * clipped with a warning.  JC2_NOLIMIT means unbounded.
 */
#define JC2_SIDE_PARAMS(X, a) \
    X(a, cjorbot,    1.0e-3,  1.0e-12, JC2_NOLIMIT, "Zero-bias capacitance per unit-of-area of bottom component") \
    X(a, cjorsti,    1.0e-9,  1.0e-18, JC2_NOLIMIT, "Zero-bias capacitance per unit-of-length of STI-edge component") \
    X(a, cjorgat,    1.0e-9,  1.0e-18, JC2_NOLIMIT, "Zero-bias capacitance per unit-of-length of gate-edge component") \
    X(a, vbirbot,    1.0,     5.0e-2,  JC2_NOLIMIT, "Built-in voltage at the reference temperature of bottom component") \
    X(a, vbirsti,    1.0,     5.0e-2,  JC2_NOLIMIT, "Built-in voltage at the reference temperature of STI-edge component") \
    X(a, vbirgat,    1.0,     5.0e-2,  JC2_NOLIMIT, "Built-in voltage at the reference temperature of gate-edge component") \
    X(a, pbot,       0.5,     5.0e-2,  0.95,              "Grading coefficient of bottom component") \
    X(a, psti,       0.5,     5.0e-2,  0.95,              "Grading coefficient of STI-edge component") \
    X(a, pgat,       0.5,     5.0e-2,  0.95,              "Grading coefficient of gate-edge component") \
    X(a, phigbot,    1.16,   -JC2_NOLIMIT, JC2_NOLIMIT, "Zero-temperature bandgap voltage of bottom component") \
    X(a, phigsti,    1.16,   -JC2_NOLIMIT, JC2_NOLIMIT, "Zero-temperature bandgap voltage of STI-edge component") \
    X(a, phiggat,    1.16,   -JC2_NOLIMIT, JC2_NOLIMIT, "Zero-temperature bandgap voltage of gate-edge component") \
    X(a, idsatrbot,  1.0e-12, 0.0,     JC2_NOLIMIT, "Saturation current density at the reference temperature of bottom component") \
    X(a, idsatrsti,  1.0e-18, 0.0,     JC2_NOLIMIT, "Saturation current density at the reference temperature of STI-edge component") \
    X(a, idsatrgat,  1.0e-18, 0.0,     JC2_NOLIMIT, "Saturation current density at the reference temperature of gate-edge component") \
    X(a, csrhbot,    1.0e2,   0.0,     JC2_NOLIMIT, "Shockley-Read-Hall prefactor of bottom component") \
    X(a, csrhsti,    1.0e-4,  0.0,     JC2_NOLIMIT, "Shockley-Read-Hall prefactor of STI-edge component") \
    X(a, csrhgat,    1.0e-4,  0.0,     JC2_NOLIMIT, "Shockley-Read-Hall prefactor of gate-edge component") \
    X(a, xjunsti,    1.0e-7,  1.0e-9,  JC2_NOLIMIT, "Junction depth of STI-edge component") \
    X(a, xjungat,    1.0e-7,  1.0e-9,  JC2_NOLIMIT, "Junction depth of gate-edge component") \
    X(a, ctatbot,    1.0e2,   0.0,     JC2_NOLIMIT, "Trap-assisted tunneling prefactor of bottom component") \
    X(a, ctatsti,    1.0e-4,  0.0,     JC2_NOLIMIT, "Trap-assisted tunneling prefactor of STI-edge component") \
    X(a, ctatgat,    1.0e-4,  0.0,     JC2_NOLIMIT, "Trap-assisted tunneling prefactor of gate-edge component") \
    X(a, mefftatbot, 0.25,    1.0e-2,  JC2_NOLIMIT, "Effective mass (in units of m0) for trap-assisted tunneling of bottom component") \
    X(a, mefftatsti, 0.25,    1.0e-2,  JC2_NOLIMIT, "Effective mass (in units of m0) for trap-assisted tunneling of STI-edge component") \
    X(a, mefftatgat, 0.25,    1.0e-2,  JC2_NOLIMIT, "Effective mass (in units of m0) for trap-assisted tunneling of gate-edge component") \
    X(a, cbbtbot,    1.0e-12, 0.0,     JC2_NOLIMIT, "Band-to-band tunneling prefactor of bottom component") \
    X(a, cbbtsti,    1.0e-18, 0.0,     JC2_NOLIMIT, "Band-to-band tunneling prefactor of STI-edge component") \
    X(a, cbbtgat,    1.0e-18, 0.0,     JC2_NOLIMIT, "Band-to-band tunneling prefactor of gate-edge component") \
    X(a, fbbtrbot,   1.0e9,  -JC2_NOLIMIT, JC2_NOLIMIT, "Normalization field at the reference temperature for band-to-band tunneling of bottom component") \
    X(a, fbbtrsti,   1.0e9,  -JC2_NOLIMIT, JC2_NOLIMIT, "Normalization field at the reference temperature for band-to-band tunneling of STI-edge component") \
    X(a, fbbtrgat,   1.0e9,  -JC2_NOLIMIT, JC2_NOLIMIT, "Normalization field at the reference temperature for band-to-band tunneling of gate-edge component") \
    X(a, stfbbtbot, -1.0e-3, -JC2_NOLIMIT, JC2_NOLIMIT, "Temperature scaling parameter for band-to-band tunneling of bottom component") \
    X(a, stfbbtsti, -1.0e-3, -JC2_NOLIMIT, JC2_NOLIMIT, "Temperature scaling parameter for band-to-band tunneling of STI-edge component") \
    X(a, stfbbtgat, -1.0e-3, -JC2_NOLIMIT, JC2_NOLIMIT, "Temperature scaling parameter for band-to-band tunneling of gate-edge component") \
    X(a, vbrbot,     10.0,    0.1,     JC2_NOLIMIT, "Breakdown voltage of bottom component") \
    X(a, vbrsti,     10.0,    0.1,     JC2_NOLIMIT, "Breakdown voltage of STI-edge component") \
    X(a, vbrgat,     10.0,    0.1,     JC2_NOLIMIT, "Breakdown voltage of gate-edge component") \
    X(a, pbrbot,     4.0,     0.1,     JC2_NOLIMIT, "Breakdown onset tuning parameter of bottom component") \
    X(a, pbrsti,     4.0,     0.1,     JC2_NOLIMIT, "Breakdown onset tuning parameter of STI-edge component") \
    X(a, pbrgat,     4.0,     0.1,     JC2_NOLIMIT, "Breakdown onset tuning parameter of gate-edge component") \
    X(a, vjunref,    2.5,     0.5,     JC2_NOLIMIT, "Typical maximum junction voltage; usually about 2*VSUP (JUNCAP-express)") \
    X(a, fjunq,      0.03,    0.0,     JC2_NOLIMIT, "Fraction below which junction capacitance components are considered negligible (JUNCAP-express)")

/* index of each per-junction parameter */
#define JC2_ENUM(a, name, def, lo, hi, desc) JC2P_##name,
enum { JC2_SIDE_PARAMS(JC2_ENUM, 0) JC2_NPAR };
#undef JC2_ENUM

/* model parameter IDs, relative to the ID base of the host model */
#define JC2_MOD_VERSION     0
#define JC2_MOD_IMAX        1
#define JC2_MOD_TRJ         2
#define JC2_MOD_FREV        3
#define JC2_MOD_SWJUNEXP    6
#define JC2_MOD_SWJUNASYM   7
#define JC2_MOD_SIDE       99      /* + side * JC2_NPAR + index */
#define JC2_MOD_NIDS       (JC2_MOD_SIDE + 2 * JC2_NPAR)

/*
 * Model parameter table entries; base is the ID base of the host model.
 *   JC2_MPTABLE(base)        common parameters and the source set
 *   JC2_MPTABLE_TREF(base)   alias TREF of TRJ, if the host has no TREF
 *   JC2_MPTABLE_DRAIN(base)  SWJUNASYM and the drain set (MOSFET only)
 */
#define JC2_IOP_S(base, name, def, lo, hi, desc) \
    IOP(#name, (base) + JC2_MOD_SIDE + JC2P_##name, IF_REAL, desc " (JUNCAP2)"),
#define JC2_IOP_D(base, name, def, lo, hi, desc) \
    IOP(#name "d", (base) + JC2_MOD_SIDE + JC2_NPAR + JC2P_##name, IF_REAL, desc " (JUNCAP2, drain)"),
#define JC2_MPTABLE(base) \
    IOP("jcap2version", (base) + JC2_MOD_VERSION,  IF_REAL, "JUNCAP2 version (only 200.6 is implemented)"), \
    IOP("imax",         (base) + JC2_MOD_IMAX,     IF_REAL, "Maximum current up to which forward current behaves exponentially (JUNCAP2)"), \
    IOP("trj",          (base) + JC2_MOD_TRJ,      IF_REAL, "Reference temperature [C] (JUNCAP2), default depends on the model"), \
    IOPR("tr",          (base) + JC2_MOD_TRJ,      IF_REAL, "Reference temperature [C] (JUNCAP2), alias of trj"), \
    IOP("frev",         (base) + JC2_MOD_FREV,     IF_REAL, "Coefficient for reverse breakdown current limitation (JUNCAP2)"), \
    IOP("swjunexp",     (base) + JC2_MOD_SWJUNEXP, IF_REAL, "Flag for JUNCAP-express; 0=full model, 1=express model"), \
    JC2_SIDE_PARAMS(JC2_IOP_S, base)
#define JC2_MPTABLE_TREF(base) \
    IOPR("tref",        (base) + JC2_MOD_TRJ,      IF_REAL, "Reference temperature [C] (JUNCAP2), alias of trj"),
#define JC2_MPTABLE_DRAIN(base) \
    IOP("swjunasym",    (base) + JC2_MOD_SWJUNASYM, IF_REAL, "Flag for asymmetric junctions (JUNCAP2); 0=drain uses source parameters, 1=drain parameters (suffix d)"), \
    JC2_SIDE_PARAMS(JC2_IOP_D, base)

/* instance parameter table entries with the instance IDs of the host model */
#define JC2_PTABLE(id_ifactor, id_cfactor) \
    IOP("ifactor",      id_ifactor, IF_REAL, "Multiplier for junction current (JUNCAP2)"), \
    IOP("cfactor",      id_cfactor, IF_REAL, "Multiplier for depletion capacitance (JUNCAP2)"),

/* temperature dependent, geometry independent values of one junction */
typedef struct {
    double P[3], one_minus_P[3], one_over_one_minus_P[3];
    double VBIR[3], VBIRinv[3], vbi[3], vbiinv[3];
    double ftd[3], idsat[3], cjo[3], qpref[3], qpref2[3];
    double wdepnulr[3], wdepnulrinv[3];
    double CSRH[3], CTAT[3], CBBT[3];
    double atat[3], btatpart[3], fbbt[3];
    double VBR[3], VBRinv[3], PBR[3], fstop[3], slope[3];
    double vjunref, fjunq;
    double par[JC2_NPAR];     /* effective (clipped) parameter values */
} JC2tvals;

/* values common to both junctions of a model */
typedef struct {
    double phitr, phitd, phitdinv;
    double perfc, berfc, cerfc;
    double alphaav;
    double imax;
    int express;
} JC2glob;

/* geometry dependent values of one junction of an instance */
typedef struct {
    double AB, LS, LG;
    double ifactor, cfactor;        /* instance parameters */
    double VMAX, exp_VMAX_over_phitd, vbimin, vch, vfmin, vbbtlim;
    /* JUNCAP-express */
    double ISATFOR1, MFOR1, ISATFOR2, MFOR2, ISATREV, MREV, m0flag;
    double xhighf1, expxhf1, xhighf2, expxhf2, xhighr, expxhr;
    double zflag[3];
} JC2inst;

/* model data, embedded in the model struct of the host model */
typedef struct {
    char *name;                     /* model name for warnings */
    double version, imax, trj, frev, swjunexp, swjunasym;
    unsigned versionGiven   :1;
    unsigned imaxGiven      :1;
    unsigned trjGiven       :1;
    unsigned frevGiven      :1;
    unsigned swjunexpGiven  :1;
    unsigned swjunasymGiven :1;
    double par[2][JC2_NPAR];        /* [0] source, [1] drain */
    unsigned char parGiven[2][JC2_NPAR];
    int warned;                     /* warnings already printed */
} JC2model;

/* temperature dependent values of a model at one temperature */
typedef struct {
    JC2glob g;
    JC2tvals t[2];                  /* [0] source, [1] drain */
} JC2tdata;

extern int  JC2mParam(JC2model *jm, int id, IFvalue *value);
extern int  JC2mAsk(const JC2model *jm, double trdef, int id, IFvalue *value);
extern void JC2setup(JC2model *jm, char *name, int active);
extern void JC2temp(JC2model *jm, JC2tdata *td, double temp, double trdef);
extern void JC2initInst(const JC2tdata *td, int side, JC2inst *in,
                        double AB, double LS, double LG, double ifactor, double cfactor);
extern void JC2eval(const JC2tdata *td, int side, const JC2inst *in, double vak,
                    double *ij, double *gj, double *qj, double *cj);

#endif
