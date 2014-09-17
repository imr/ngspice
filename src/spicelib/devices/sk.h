/*-----------------------------------------------------------------------------
Do not change this data manually,
it will be automatically updated by RCS

$RCSfile$
------------------------------------------------------------------------------*/
/*
 * Copyright (c) 2002 - 2012 NXP Semiconductors
 * Eindhoven, The Netherlands
 */

/**
 * @file
 * Interface file for the SiMKit device model library.
 *
 * This file declares the data structures of the SiMKit model interface.
 * The core of the interface consists of four descriptors containing data
 * fields and function pointers:
 * The SK_MODELLIB_DESCRIPTOR is the entry point for the model library
 * of the SiMKit. It contains functions to get information about the library
 * version and its interface and functions to get access to the device models
 * contained by the model library. Access to the single instantiation of
 * this descriptor is possible via the symbol: "SK_modellib_descriptor" which
 * is the only exported symbol of the SiMKit model library.
 * The SK_DEVICE_DESCRIPTOR contains generic data and functions valid for a
 * specific device. A device is a mathematical description of a transistor
 * or other electronic element. The device descriptor mainly contains
 * the descriptions of the parameters for the mathematical expressions.
 * The device descriptor does not contain any actual parameters values.
 * A SK_MODEL_DATA object contains a set of model parameter values. It
 * is an abstract concept, allowing grouping of device instances on the
 * basis of common parameter settings for the mathematical description.
 * A SK_INSTANCE_DATA object corresponds with an actual instance in a
 * netlist. It contains the values for instance specific parameters and
 * provides functions for evaluating the mathematical expressions.
 */

#ifndef _SK_H
#define _SK_H

#ifndef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#endif

/******************************************************************************
 *                        I N C L U D E   F I L E S                           *
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#ifndef _WIN32
#include <ctype.h>
#endif
#include <errno.h>
#include <stdlib.h>

/******************************************************************************
 *                       SiMKit device interface version number               *
 ******************************************************************************/

/**
 * The SiMKit device interface version number.
 * Any time the interface changes this value has to be increased.
 * The SiMKit adapters for Pstar, Spectre and ADS use this define also
 * to check whether their code supports the interface version the compiled
 * device library reports.
 *
 * Version 1: February 2007 The first numbered version of the SK interface.
 *
 * Version 2: February 2007 Added parameter 'p_is_param_changed' to
 *            set_model_params and to set_inst_params (ER18977)
 *            Added p_ev_ig_available array to SK_MODEL (ER19057).
 *
 * Version 3: June 2007 Restructuring of the interface. Moved all instance
 *            specific data (mainly topology info) to the new SK_INSTANCE_DATA
 *            struct. Added the SK_MODEL_DATA struct.
 *
 * Version 4: January 2008 (ER19420, phase3) Removed p_noise_source_status field
 *            from SK_INSTANCE_DATA struct. Removed SK_STATUS_TYPE.
 *            February 2008 (ER19446, phase1) Extended SK_MODEL_DATA with
 *            function 'p_get_model_type' to retrieve the model type (gender).
 *            Changed 'p_set_defaults_for_type' into 'p_set_defaults_and_model_type'.
 *            Renamed 'p_foreign_h_index' to 'p_foreign_h_nrs'.
 *
 * Version 5: August 2008 (ER19522) Extended the SK_INSTANCE_DATA struct with the
 *            new function  'get_topology_id' to make it possible to check if the
 *            topologies used by two device-instances are the same.
 *            August 2008 (ER19587) Extended the SK_SIM_DATA struct with
 *            'row_scale_factor' needed for simulator dependent numerical scaling
 *             of some equations.
 *            October 2008 Added 'p_name' to the SK_MODEL_DATA struct and as
 *            argument of function 'p_create_model_data'.
 *
 * Version 6: May 2009 (ER19824) Extended typedef SK_DEVICE_FAMILY with a
 *            new family SK_DEVICE_FAMILY_PSP to make it possible to implement
 *            e.g. overvoltage checks generically for MOS and PSP devices.
 *
 * Version 7: August 2011 (ER20523) Addded struct SK_MODELLIB_DESCRIPTOR which is
 *            the new entry point for the model library of the SiMKit. It contains
 *            functions to get information about the library version and its
 *            interface and functions to get access to the device models contained
 *            by the model library.
 *            The name of the exported symbol for the model library descriptor is:
 *            "SK_modellib_descriptor".
 *            The old (deprecated) library access functions are still available
 *            for backwards compatibility. The function prototypes are added to
 *            sk.h.
 *
 * Version 8: May 2012 (ER20093) Completely revised interface, incompatible with
 *            the previous interface versions. This simkit interface is fully
 *            thread-safe. See simkitInterfaceDescription.docx for details.
 */
#define SK_INTERFACE_VERSION            8

/* NOTE FOR SIMKIT DEVELOPERS:
 * When increasing SK_INTERFACE_VERSION, check if the lower boundary setting
 * for the backward compatibility (SK_MIN_ADAPTER_INTERFACE_VERSION) defined in
 * modellib_descriptor.c is still valid.
 */

/******************************************************************************
 *                               M A C R O S                                  *
 ******************************************************************************/

/**
 * Node number of a non-existing node.
 * Needed to handle branches that have only one node (refer to: SK_NODE and SK_BRANCH)
 */
#define SK_NODE_NUMBER_NONE    -1

/**
 * Topology ID for fixed topology devices
 */
#define SK_TOPOLOGY_FIXED       0xFFFF


/**
 * Maximum values needed for allocation of arrays in non-C99 mode
 */
#define N_MAX_TERMINALS         40      /**< Maximum possible value of n_max_terminals */
#define N_MAX_INT_NODES         40      /**< Maximum possible value of n_max_int_nodes */
#define N_MAX_EVS               65      /**< Maximum possible value of n_max_evs */
#define N_MAX_BRANCHES          80      /**< Maximum possible value of n_max_branches */
#define N_MAX_NOISE_SOURCES     30      /**< Maximum possible value of n_max_noise_sources */
#define N_MAX_FOREIGN_H         2       /**< Maximum possible value of n_foreign_h */


/*----------------------------------------------------------------------------*
 *  Fundamental Physical Constants                                            *
 *  source: Handbook of Chemistry and Physics, 54th edition, 1973             *
 *----------------------------------------------------------------------------*/

#define PI          3.14159265358979323844  /**< pi, 21 decimal digits */
#define TWOPI       6.28318530717958647688  /**< 2*pi, 21 decimal digits */
#define SQRT2       1.41421356237309504880  /**< sqrt(pi), 21 decimal digits */
#define DEGPERRAD   57.29577951308232087721 /**< Degrees per radian, 22 decimal digits*/
#define Q_ELECTRON  1.6021918E-19           /**< Electron charge (C) */
#define K_BOLTZMANN 1.3806226E-23           /**< Boltzmann's constant (J/K)*/
#define EPSILON_0   8.854214871e-12         /**< Dielectric constant (F/m) */
#define EPSILON_OX  3.453143800e-11         /**< Permittivity of SiO2 (F/m) */
#define EPSILON_SI  (11.7*EPSILON_0)        /**< Dielectric constant of silicon (F/m)*/
#define KELVIN_CONVERSION 273.15            /**< Celsius to Kelvin conversion (K) */

/* Derived constant */
#define K_B_div_Q_EL 8.61708691805812512584e-5 /**< K_BOLTZMANN / Q_ELECTRON */


/*----------------------------------------------------------------------------*
 *           M I S C E L L A N E O U S     M A C R O S                        *
 *----------------------------------------------------------------------------*/

#define UNDEF_NUMBER    1.654321E38         /**< Undefined number */

/* Macro to get rid of GCC's unused parameter warning. When using the GCC
 * compiler this applies the unused attribute and also mangles the variable
 * name so that you really can't use it. */
#ifdef __GNUC__
#define SK_UNUSED(x) UNUSED_ ## x __attribute__((unused))
#else
#define SK_UNUSED(x) x
#endif


/******************************************************************************
 *                              TYPE DEFINITIONS                              *
 ******************************************************************************/

/* SiMKit general types. */
typedef double                          sk_real;
typedef int                             sk_integer;
typedef unsigned int                    sk_unint;
typedef char                            sk_boolean;

/* Defines for boolean values. */
#ifndef FALSE
#define FALSE                           0
#define TRUE                            1
#endif


/**
 * Complex data type. */
typedef struct sk_complex
{
  sk_real       Real;                   /**< Real part */
  sk_real       Imag;                   /**< Imaginary part */
} sk_complex;


/**
 * General mask type. */
typedef unsigned int                    SK_MASK_TYPE;

/**
 * Device info flags
 * Returned by p_get_device_info to give (static) information on the device */
typedef SK_MASK_TYPE SK_DEVICE_INFO;
#define SK_DI_HAS_STATIC_BRANCHES       0x0001      /**< device has static branches */
#define SK_DI_HAS_DYNAMIC_BRANCHES      0x0002      /**< device has dynamic branches */
#define SK_DI_HAS_NONLINEAR_BRANCHES    0x0004      /**< device has non-linear branches */
#define SK_DI_IS_NOT_THREAD_SAFE        0x0010      /**< implementation of device is not thread-safe */
#define SK_DI_FLEXIBLE_TOPOLOGY         0x0020      /**< device has a flexible topology */
#define SK_DI_SELF_HEATING              0x0040      /**< device is a self-heating device */

/**
 * Model type flags
 * The default option is used if the model type is not known yet
 * in which case the model selects its own default (N, P, or NONE) */
typedef SK_MASK_TYPE SK_MODEL_TYPE;
#define SK_MT_N                         0x0001      /**< N type */
#define SK_MT_P                         0x0002      /**< P type */
#define SK_MT_NONE                      0x0004      /**< No type */
#define SK_MT_DEFAULT                   0x0008      /**< Default type, the model itself chooses N, P, or NONE */

/**
 * Initialisation control flags mask. 
 * NOTE: these flags are only for backwards compatibility
 *       They are not used anymore since simkit version 4.1
 */
typedef SK_MASK_TYPE SK_INIT_CONTROL;
#define SK_IC_NONE                      0x0000
#define SK_IC_EVAL_COMMON_PARAMS        0x0001
#define SK_IC_GEOM_SCALING              0x0002
#define SK_IC_TEMP_SCALING              0x0004
#define SK_IC_MULT_SCALING              0x0008
#define SK_IC_ALL_SCALING               0x000F

/**
 * Evaluation control flags mask. */
typedef SK_MASK_TYPE SK_EVAL_CONTROL;
#define SK_EC_NONE                      0x0000
#define SK_EC_CURRENTS                  0x0001
#define SK_EC_CHARGES                   0x0002
#define SK_EC_CONDUCTANCES              0x0004
#define SK_EC_CAPACITANCES              0x0008
#define SK_EC_ALL_OUTPUT_EVS            0x000F

/**
 * Parameter type flags mask. */
typedef SK_MASK_TYPE SK_PARAM_TYPE;
#define SK_PT_NONE                      0x0000
#define SK_PT_MAXISET_MODEL             0x0001
#define SK_PT_MAXISET_INST              0x0002
#define SK_PT_MINISET                   0x0004
#define SK_PT_ELECTRICALSET             0x0008
#define SK_PT_OPERATINGPOINT            0x0010

/**
 * Parameter clipping type flags mask. */
typedef SK_MASK_TYPE SK_CLIP_TYPE;
#define SK_CB_NONE                      0x0000
#define SK_CB_LOWER                     0x0001
#define SK_CB_UPPER                     0x0010
#define SK_CB_BOTH                      0x0011

/**
 * Parameter dependency type flags mask. */
typedef SK_MASK_TYPE SK_AFFECTS_TYPE;
#define SK_AT_AFFECTS_NONE              0x0000
#define SK_AT_TEMP_AFFECTS              0x0001
#define SK_AT_AFFECTS_OP_POINT          0x0002
#define SK_AT_OP_POINT_AFFECTS          0x0004


/**
 * Parameter scaling types to support the Spectre 'scale' parameter.
 * Maxiset instance parameters that have a length in their unit, are scaled.
 * For example, the value of a parameter with unit "m^2" is
 * multiplied by factor^2. This also goes for complex units, e.g. m^2/V.
 */
typedef enum
{
    SK_SF_NONE,                         /**< no length unit */
    SK_SF_DIVIDE,                       /**< 1/m   */
    SK_SF_POW_MIN_TWO,                  /**< 1/m^2 */
    SK_SF_POW_MIN_THREE,                /**< 1/m^3 */
    SK_SF_POW_MIN_FOUR,                 /**< 1/m^4 */
    SK_SF_MULTIPLY,                     /**< m     */
    SK_SF_POW_TWO,                      /**< m^2   */
    SK_SF_POW_THREE,                    /**< m^3   */
    SK_SF_POW_FOUR                      /**< m^4   */
} SK_SCALING_FACTOR;


/**
 * Node types.
 */
typedef enum
{
    SK_ND_NONE,                         /**< Node type is undefined. */
    SK_ND_TERMINAL,                     /**< Node is a terminal node of the device. */
    SK_ND_INTERNAL_NODE                 /**< Node is an internal node of the device. */
} SK_NODE_TYPE;


/**
 * Branch types.
 */
typedef enum
{
    SK_BT_ELASTIC,          /**< Elastic branch (static branch belonging to DC path). */
    SK_BT_NONELASTIC,       /**< Nonelastic branch (static branch not belonging to DC path). */
    SK_BT_DYNAMIC,          /**< Dynamic branch. */
    SK_BT_GMIN,             /**< Gmin branch. */
    SK_BT_DIDVMIN           /**< Didvmin branch. */
} SK_BRANCH_TYPE;


/**
 * Noise types
 */
typedef enum
{
    SK_NT_THERMAL,          /**< noise is neither a function of frequency nor time varying. */
    SK_NT_SHOT,             /**< noise is time varying, but not a function of frequency. */
    SK_NT_FLICKER,          /**< noise = |h(t)|/frequency. */
    SK_NT_UNCORR,           /**< general case where noise = |h(t)|*|U(|h(t)|,f)|. */
    SK_NT_CORR_REAL,        /**< general case of real correlated sources. */
    SK_NT_CORR_IMAG         /**< general case of imag correlated sources. */
} SK_NOISE_TYPE;


/**
 * SiMKit error codes.
 */
typedef enum
{
    SK_ERR_NONE,                         /**< No error, all is OK. */
    SK_ERR_PARAMETER_NOT_READABLE,       /**< Trying to get a write-only parameter. */
    SK_ERR_PARAMETER_NOT_WRITEABLE,      /**< Trying to set a read-only parameter. */
    SK_ERR_UNKNOWN_PARAMETER,            /**< Specified an unknown parameter ID when getting or setting parameters. */
    SK_ERR_UNKNOWN_TYPE_STRING,          /**< Specified an unknown type ID (P or N). */
    SK_ERR_LIMITED,                      /**< Warning that values have been limited/clipped in diode evaluation. */
    SK_ERR_CHANGE_TERMINALS_NOT_ALLOWED, /**< Repeated call to set_n_terminals() with different number of terminals. */
    SK_ERR_INVALID_NUMBER_OF_TERMINALS,  /**< The specified number of terminals is not supported by the device. */
    SK_ERR_PARAMS_CHANGE_TOPOLOGY,       /**< Setting model/instance parameters would change a frozen topology. */
    SK_ERR_OUT_OF_MEMORY,                /**< Out of memory. */
    SK_ERR_ERROR                         /**< Generic error message. */
} SK_ERROR;


/*----------------------------------------------------------------------------*
 * Interface that the models use to report info / warnings / errors to the    *
 * simulator (as opposed to the adapter, which uses SK_ERROR).                *
 *----------------------------------------------------------------------------*/

/**
 * Message codes sent by the models to the adapter to report an error, a warning or an
 * info message to the simulator. The adapter converts this information to simulator
 * specific status reporting. */
typedef enum
{
    SK_REP_NONE,
    SK_REP_COPYRIGHT_NOTICE,            /**< Copyright notice. */
    SK_REP_PARAMETER_CLIPPED,           /**< Parameter clip warning. */
    SK_REP_NEGATIVE_WEFF,               /**< Effective width becomes negative. */
    SK_REP_NEGATIVE_LEFF,               /**< Effective length becomes negative. */
    SK_REP_INFO_TEXT,                   /**< Debug output. */
    SK_REP_PARAMETER_READONLY,          /**< Attempting to set read-only parameter. */
    SK_REP_PARAMETER_UNKNOWN,           /**< Attempting to set unknown parameter. */
    SK_REP_PARAMETER_CHANGES_TOPOLOGY,  /**< Setting this parameter would change the frozen topology. */
    SK_REP_SIMKIT_MESSAGE,              /**< SiMKit adapter and model version message. */
    SK_REP_DEVICE_TOO_HOT,              /**< Selfheating device becomes too hot. */
    SK_REP_OUT_OF_MEMORY,
    SK_REP_FAILED_GEOM_CHECK
#ifdef SPICE3
    ,SK_REP_SPICE3_WARNING              /**< Warning. */
    ,SK_REP_SPICE3_ERROR                /**< Error. */
    ,SK_REP_SPICE3_FATAL                /**< Fatal error. */
#endif /* SPICE 3 */
    ,SK_REP_UNKNOWN_TYPE_STRING         /**< Unknown gender for this model. */
} SK_REPORT_STATUS;

/**
 * Type of the function to report warnings, errors and debug output in a simulator
 * specific way. This function has to be implemented by the adapter and may be called by
 * any model or by the adapter itself.
 * @param aStatus Message code
 * @param anInfo  Strings to be put in the message
 */
typedef void (*SK_REPORT_TO_SIMULATOR_FUNC)(SK_REPORT_STATUS aStatus, const char* anInfo[]);

/**
 * Maximum number of strings to be put in the message that is sent by SK_REPORT_TO_SIMULATOR_FUNC
 * to the adapter.
 */
#define SK_REPORT_INFO_SIZE         6


/*-------------------------- Device family types -----------------------------*/

/**
 * Family of devices. It is used by the adapters to determine if a device needs
 * special processing such as limiting (for the bipolar family devices) or special
 * source-drain interchange processing (for the MOST family devices).
 */
typedef enum
{
    /**
     * Device family that contains MOST devices that need most
     * source-drain limiting.
     */
    SK_DEVICE_FAMILY_MOST,

    /**
     * Device family that contains bipolar devices that need junction
     * limiting; when this family is set, p_limit_info must be
     * set to a valid limiting info structure.
     */
    SK_DEVICE_FAMILY_BJT,

    /**
     * Device family that contains PSP devices that are very similar to
     * most devices.
     */
    SK_DEVICE_FAMILY_PSP,

    /**
     * Device family that contains all other device models; no
     * limiting or other special processing is performed for these
     * devices.
     */
    SK_DEVICE_FAMILY_GENERIC

} SK_DEVICE_FAMILY;


/*------------------------- Simulation data structure ------------------------*/

/**
 * Structure to transfer simulation specific settings to the model
 */
typedef struct SK_SIM_DATA
{
    /**
     * Global instance scaling factor in Spectre used in
     * clip_and_scale_params() and in get_inst_params().
     * Default value: 1.0 (means: no-effect)
     */
    sk_real                       inst_scale;

    /**
     * Factor used for simulator dependent scaling of some supplemental
     * equations to improve numerical stability used in eval_model().
     * Default value: 1.0 (means: no-effect)
     */
    sk_real                       row_scale_factor;

} SK_SIM_DATA;


/*------------------------ Initial guess definitions -------------------------*/

/**
 * Identifiers for initial guess values for EVs. Each value represents a device
 * state. Not all states may apply to all types of devices.
 */
typedef enum
{
    /* Spectre */
    SK_IG_OFF_SPECTRE,      /**< initial guess for OFF device state (Spectre) */
    SK_IG_TRIODE,           /**< initial guess for TRIODE device state (Spectre) */
    SK_IG_SATURATION,       /**< initial guess for SATURATION device state (Spectre) */
    SK_IG_SUBTHRESHOLD,     /**< initial guess for SUBTHRESHOLD device state (Spectre) */
    SK_IG_REVERSE,          /**< initial guess for REVERSE device state (Spectre) */
    SK_IG_FORWARD,          /**< initial guess for FORWARD device state (Spectre) */
    SK_IG_BREAKDOWN,        /**< initial guess for BREAKDOWN device state (Spectre) */

    /* Pstar */
    SK_IG_DEFAULT,          /**< initial guess for DEFAULT device state */
    SK_IG_ON,               /**< initial guess for ON device state */
    SK_IG_OFF_PSTAR,        /**< initial guess for OFF device state (PStar) */

    SK_IG_N_REGION          /**< number of initial guess regions */
} SK_INITIAL_GUESS_TYPE;

/**
 * Type of initial guess values of EVs
 */
typedef sk_real                   SK_INITIAL_EV;


/*---------------------------- Region definitions ----------------------------*/

/**
 *  Device working region for MOST devices.
 *  A circuit simulator might make use of the working region for efficiency
 *  purposes.
 */
typedef enum
{
    MOS_OFF,
    MOS_SATURATION,
    MOS_TRIODE,
    MOS_SUBTHRESHOLD
} SK_MOS_REGION;

/**
 *  Device working region for diode devices.
 *  A circuit simulator might make use of the working region for efficiency
 *  purposes.
 */
typedef enum{
    DIODE_OFF,
    DIODE_ON,
    DIODE_BKDN
} SK_DIODE_REGION;


/*--------------------------- Limiting definitions ---------------------------*/

/**
 * Limiting types
 */
typedef enum
{
    SK_LIM_MOS,             /**< Limiting for MOS transistors. */
    SK_LIM_BJT              /**< Limiting for bipolar transistors. */
} SK_LIM_TYPE;


/*--------------------------- Parameter descriptor ---------------------------*/

/**
 * Parameter descriptor structure. This structure contains
 * the definition of a parameter including clipping and default
 * values but not its actual value.
 * All parameters (model, instance and OP parameters) are defined
 * in a single list. Each parameter has a unique ID number.
 * The model takes care of when to clip what parameter, the adapter has nothing
 * to do with this and does not use this information.
 */
typedef struct SK_PARAM_DESCRIPTOR
{
    /**
     * ID number, used to identify parameters in the access functions. This number
     * must be unique in the set of all parameters for a model. The numbering of
     * parameters must start with 0, be consecutive and number the model
     * parameters, instance parameters and operating point parameters of the model,
     * in that order. It is suggested to use an enum to create the parameter numbers.
     * The parameter numbers determine the order of the parameters in the
     * SK_DEVICE_DESCRIPTOR pp_params array.
     */
    sk_unint                      number;

    /**
     * String containing the name of the parameter. White spaces, tabs, new lines etc.
     * are not allowed and, by convention, parameter names are upper case only (to
     * be converted to the correct case by the simulator specific adapters). Parameter
     * names should be unique.
     */
          char                   *p_name;
    const char                   *p_description;  /**< String describing the parameter. */
    const char                   *p_unit;         /**< A string that represents the unit of the parameter. */
          sk_boolean              is_readable;    /**< If this flag is set to TRUE , the value of the parameter can be retrieved with the
                                                       function get_model_params() respectively get_instance_params() by the
                                                       adapter. It is set to FALSE only for miniset parameters. */
          sk_boolean              is_writeable;   /**< If this flag is set to TRUE , the value of the parameter can be set with the
                                                       function set_model_params() respectively set_instance_params() by the
                                                       adapter. Only parameters that can be set in the netlist (the maxiset) may be writeable. */
          sk_boolean              is_model_param; /**< If this flag is set to TRUE , the parameter is a maxiset model parameter. */
          SK_SCALING_FACTOR       scaling_type;   /**< The scaling rule that is applied to the parameter. */
          SK_CLIP_TYPE            clip_type_n;    /**< Flag that specifies for an n-type of device which clipping bounds are to be
                                                       used. When set to SK_CB_NONE , set the clip_low_n and clip_high_n values to SK_NOT_USED . */
          SK_CLIP_TYPE            clip_type_p;    /**< Flag that specifies for an p-type of device which clipping bounds are to be
                                                       used . When set to SK_CB_NONE , set the clip_low_p and clip_high_p values to SK_NOT_USED . */
          sk_real                 clip_low_n;     /**< Lower clipping bound for an n-type device (if the parameter value is smaller than this value, the parameter value is set to the clip_low_n value). */
          sk_real                 clip_low_p;     /**< Lower clipping bound for a p-type device (if the parameter value is smaller than this value, the parameter value is set to the clip_low_p value). */
          sk_real                 clip_high_n;    /**< Upper clipping bound for an n-type device (if the parameter value is greater than this value, the parameter value is set to the clip_high_n value). */
          sk_real                 clip_high_p;    /**< Upper clipping bound for a p-type device (if the parameter value is greater than this value, the parameter value is set to the clip_high_p value). */
          sk_real                 default_value_n;/**< Default value for an N-type device. */
          sk_real                 default_value_p;/**< Default value for an P-type device. */

    /**
     * Flag that indicates the dependency of this parameter on temperature and operating point (and vv).
     * The flag can have the following values (OR-operation is possible):
     * SK_AT_TEMP_AFFECTS Indicates that this parameter is a direct function of temperature. It is typically
     * used to determine which parameters to print when printing parameters versus
     * temperature, and to determine which parameters to recalculate when the
     * device temperature changes. Set only for the electrical set.
     * SK_AT_AFFECTS_OP_POINT Indicates that changing this parameter would change the DC solution of the
     * circuit. When a parameter with this flag set is modified, the DC solution may
     * be recomputed automatically by the simulator.
     * SK_AT_OP_POINT_AFFECTS Indicates that changing the operating point (DC solution) would change the
     * value of this parameter [e.g. operating point information]. It also implies that
     * the DC solution must be known and available, in order to compute and output this parameter.
     */
          SK_AFFECTS_TYPE         dependency;
          SK_PARAM_TYPE           param_type;     /**< This indicates the parameter type. The type is used by the generic parameter clipping functions in the SiMKit. */
} SK_PARAM_DESCRIPTOR;


/* parameter descriptor define for unused fields */
#define SK_NOT_USED               0


/*--------------------------- SiMKit basic objects ---------------------------*/

/**
 * This structure describes a node of a device model (both internal as
 * well as terminal nodes).
 * Each node in a model has a unique number, a name and a type (internal
 * or terminal node).
 * In the adapters, an internal node number of ' SK_NODE_NUMBER_NONE ' is used during
 * processing to denote a non-existing node e.g. in case of a branch that has only one
 * node. In shared_device_code.c, a SK_NODE node_none is defined that contains this
 * SK_NODE_NUMBER_NONE and which must be used by the models to denote a non-existing
 * node.
 */
typedef struct SK_NODE
{
    /**
     * Number of the node. Nodes must be numbered starting 0 and then
     * consecutively numbering the terminals and the internal nodes, in that order.
     * The node numbers are unique, topology independent and will never change.
     * The node numbers determine the order of the nodes in the pp_all_term_nodes
     * and pp_all_int_nodes arrays and belong to the maximum topology description.
     */
    sk_integer              number;
    char                   *p_name;     /**< Name of the node. Convention is to use all capital letters. */
    SK_NODE_TYPE            type;       /**< Type of the node. */
} SK_NODE;


/**
 * This structure describes a branch in a device model. A branch is
 * a connection between two nodes. A model computes currents, voltages,
 * conductances and charges across or on branches. The values on a branch
 * are determined by a set of controlling electrical variables (EVs).
 * The numbers of the controlling EVs of a branch are stored in the
 * branch.
 * The array of controlling EVs can be used to optimize matrix insertion: only
 * derivatives with respect to these EVs are inserted into the matrices.
 */
typedef struct SK_BRANCH
{
    /**
     * Number of the branch. Branches must be numbered starting at 0 and then
     * consecutively numbering the static, static linear, dynamic and dynamic linear
     * branches, in that order. The branch numbers are unique, topology independent
     * and will never change.
     * Gmin branches are numbered separately and consecutively, starting at 0.
     * Didvmin branches are not defined separately as they depend on the current
     * through another branch: the branch definition of this branch is re-used.
     */
    sk_unint                number;
    char                   *p_name;          /**< Name of the branch. Convention is to use all capitals. */
    SK_NODE                *p_pos_node;      /**< Positive node of the branch. */
    SK_NODE                *p_neg_node;      /**< Negative node of the branch, or, if the branch has only one node
                                                  (such as a current source) node_none. node_none is a globally defined node
                                                  structure that contains a node number SK_NODE_NUMBER_NONE. The adapter
                                                  may use this number to check if the node exists or not. */
    sk_unint                n_ctrl_ev;       /**< Number of electrical variables (EVs) that influence the state of the branch. */
    sk_unint               *p_ctrl_ev_nrs;   /**< Array of controlling EV numbers that influence the state of the branch.
                                                  The size of the array must be n_ctrl_ev. */
    SK_BRANCH_TYPE          type;            /**< Type of the branch (to help convergence). */
} SK_BRANCH;

/**
 * The noise source descriptor contains information about the type of noise source and to
 * what nodes in the equivalent circuit it is related. This information is used by the
 * adapter to determine how to calculate the total noise density correctly from the noise
 * parts calculated by the device model.
 * The spectral current density is given by noise = |h(t)|*|U(|h|,f)|.
 * So for each source there is a bias dep part h(t) and a  freq dep part u(|h|,f).
 * If a u-function also depends on h-functions of other sources, then the p_foreign_h_nrs
 * array is used to store the source numbers of these noise sources. The p_foreign_h_nrs
 * array is set in the model code.
 * The p_foreign_h_nrs can be used for all noise types (SK_NOISE_TYPE).
 * The nodes p_pos/neg_node_corr are used to model correlated noise sources.
 */
typedef struct SK_NOISE_SOURCE
{
    sk_unint                number;             /**< Number of the noise source. Noise sources must be numbered starting at 0,
                                                     consecutively. The noise numbers are unique, topology independent and will never change. */
    char                   *p_name;             /**< Name of the noise source. */
    SK_NOISE_TYPE           type;               /**< Type of the noise source. Used to distinguish different types of correlated and uncorrelated noise. */
    SK_NODE                *p_pos_node;         /**< The positive node of the branch the noise is associated with. */
    SK_NODE                *p_neg_node;         /**< The negative node of the branch the noise is associated with. */
    SK_NODE                *p_pos_node_corr;    /**< The positive node of the branch the noise is correlated with, or if the noise is uncorrelated NULL. */
    SK_NODE                *p_neg_node_corr;    /**< The negative node of the branch the noise is correlated with, or if the noise is uncorrelated NULL. */
    sk_unint                n_foreign_h;        /**< Number of foreign h-functions, influencing the u-function of this noise source. */
    sk_unint               *p_foreign_h_nrs;    /**< Array of noise source numbers influencing the u-function of this noise source. */
} SK_NOISE_SOURCE;


/*--------- EV descriptor, input EV, output EV, bias dependent info  ---------*/

/**
 * The EV descriptor is used to describe an electrical variable that is
 * an unknown of the model the simulator is trying to solve. The model
 * uses these as input for calculations.
 */
typedef struct SK_EV_DESCRIP
{
    sk_unint                      number;          /**< Number of the EV. EV numbers must be numbered starting at 0, consecutively.
                                                        The EV numbers are unique, topology independent and will never change. */
    char                         *p_name;          /**< Name of the EV. Convention is to use all capitals */
    SK_INITIAL_EV          const *p_initial_value; /**< Array of initial values for the EV */
    SK_NODE                      *p_pos_node;      /**< Positive node the EV is associated with */
    SK_NODE                      *p_neg_node;      /**< Negative node the EV is associated with, or, if the EV has only one node
                                                        (such as a temperature EV) node_none. node_none is a globally defined node
                                                        structure that contains a node number SK_NODE_NUMBER_NONE. The adapter
                                                        may use this number to check if the node exists or not. */
} SK_EV_DESCRIP;

/**
 * An input EV contains the latest values the simulator calculated for the unknowns
 * of the device.
 */
typedef struct SK_INPUT_EV
{
    sk_real                       value;               /**< The value of the EV. */
    sk_real                       previous_value;      /**< The value of the EV during the previous iteration. This may be used by limiting algorithms. */
    sk_boolean                    is_initial_guess;    /**< Shows whether the value of the EV is based on an initial guess */
} SK_INPUT_EV;

/**
 * An output EV contains electrical quantities (currents, charges) and their
 * derivatives for a single branch of the device.
 */
typedef struct SK_OUTPUT_EV
{
    sk_real                       value;           /**< The calculated electrical quantity. */
    sk_real                      *p_derivatives;   /**< Array (size n_max_evs) containing the derivatives of the calculated quantity with respect to each input EV. */
} SK_OUTPUT_EV;

/**
 * The bias_dep_info contains bias dependent quantities that are output
 * of eval_model (next to the output ev's).
 */
typedef struct SK_BIAS_DEP_INFO
{
    SK_MOS_REGION                  mos_region;     /**< Working region of mos device */
    SK_DIODE_REGION                diode_region;   /**< Working region of diode */
    sk_boolean                     imax_exceeded;  /**< Indicates if a junction current exceeds imax  */
    sk_boolean                     insert_didvmin; /**< Set by the device implementation to inform the adapters of didvmin insertion */
    sk_boolean                     inverse_function_applied; /**< Indicates whether any bias values were limited. */
    sk_boolean                     dc_or_disc_timestep;  /**< Flag to indicate a DC timestep or a transient timestep with discontinuity (needed for Pstar). */
    sk_unint                       iter_count;    /**< Iteration number. Set by the adapter. */
    sk_real                       *p_truncated_ev_values; /**< Truncated evs (size n_max_evs or NULL if not used) */
} SK_BIAS_DEP_INFO;


/*-------------------------------- Limiting ----------------------------------*/

/**
 * Limiting for diode models and selfheating
 *     p_eval_diode calculates the diode current Id = ISt * (exp(Vd_nvt) - 1.0)
 *     p_eval_exp calculates a limited exponential to prevent overflow
 *     p_eval_v_check calculates the voltage above which the diode current exceeds Imax
 *     p_limit_temperature limits the temperature in case of self-heating
 */
typedef struct SK_LIMITING
{
    SK_LIM_TYPE                   limiting_type;      /**< The type of limiting used by the device. */

    SK_ERROR (*p_eval_diode) (
        sk_real                   ISt,                /**< ISt */
        sk_real                  *p_Vd,               /**< Voltage across diode */
        sk_real                   Vd_prev,            /**< Previous voltage across diode */
        sk_real                   Vt,                 /**< K*T/q, the thermal voltage */
        sk_real                  *p_rhs,              /**< Right hand side (Pstar specific) */
        sk_real                  *p_jac,              /**< Derivative of diode current w.r.t. *p_Vd */
        sk_real                   v_check,            /**< voltage above which the exponential is linearised */
        sk_real                   min_jac,            /**< Minimal value of *p_jac (Pstar specific) */
        sk_boolean                is_initial_guess,   /**< True if *p_Vd is initial guess */
        sk_real                  *p_value,            /**< The diode current */
        SK_BIAS_DEP_INFO         *p_bias_dep_info );  /**< Bias dependent info */

    SK_ERROR (*p_eval_exp) (
        sk_real                   x,                  /**< Argument of exponential */
        sk_real                   x0,                 /**< Value above which the exponential is linearised */
        sk_real                  *p_value,            /**< The (linearised) exponential */
        sk_real                  *p_dvalue   );       /**< The derivative of the (linearised) exponential w.r.t. x */

    SK_ERROR (*p_eval_v_check) (
        sk_real                   ISt,                /**< ISt */
        sk_real                   Imax,               /**< Imax */
        sk_real                   Vt,                 /**< K*T/q, the thermal voltage */
        sk_real                  *p_v_check   );      /**< The voltage where Idiode == Imax */

    SK_ERROR (*p_limit_temperature) (
        sk_real                  *p_T,                /**< Current temperature value */
        sk_real                   T_prev,             /**< Previous temperature value */
        sk_boolean                is_initial_guess,   /**< True if *p_T is initial guess */
        SK_BIAS_DEP_INFO         *p_bias_dep_info );  /**< Bias dependent info */

} SK_LIMITING;


/*--------------------------- Topology specific data -------------------------*/

/**
 * This structure contains all topology related elements in SiMKit devices.
 */
typedef struct SK_TOPOLOGY {
    sk_unint          n_terminals;              /**< Number of terminals */
    SK_NODE         **pp_terminals;             /**< List of terminals, length n_terminals */

    sk_unint          n_int_nodes;              /**< Number of internal nodes */
    SK_NODE         **pp_int_nodes;             /**< List of internal nodes, length n_int_nodes */

    sk_unint          n_static_branches;        /**< Number of static branches */
    SK_BRANCH       **pp_static_branches;       /**< List of static branches, length n_static_branches */

    sk_unint          n_static_linear_branches; /**< Number of static linear branches */
    SK_BRANCH       **pp_static_linear_branches;/**< List of static linear branches, length n_static_linear_branches */

    sk_unint          n_dynamic_branches;       /**< Number of dynamic branches */
    SK_BRANCH       **pp_dynamic_branches;      /**< list of dynamic branches, length n_dynamic_branches */

    sk_unint          n_dynamic_linear_branches;/**< Number of dynamic linear branches */
    SK_BRANCH       **pp_dynamic_linear_branches;/**< List of dynamic linear branches, length n_dynamic_linear_branches */

    sk_unint          n_gmin_branches;          /**< Number of Gmin branches */
    SK_BRANCH       **pp_gmin_branches;         /**< List of Gmin branches, length n_gmin_branches */

    sk_unint          n_didvmin_branches;       /**< Number of dI/dV min branches */
    SK_BRANCH       **pp_didvmin_branches;      /**< List of dI/dV min branches, length n_didvmin_branches */

    sk_unint          n_noise_sources;          /**< Number of noise sources */
    SK_NOISE_SOURCE **pp_noise_sources;         /**< List of noise sources, length n_noise_sources */

    sk_unint          n_evs;                    /**< Number of input EVs available in this topology */
    SK_EV_DESCRIP   **pp_input_ev_descrips;     /**< List of input EV descriptors, length n_evs */

    sk_unint          n_max_evs;                /**< The maximum number of EVs (is the largest possible hard-coded EV-number) */
    SK_EV_DESCRIP   **pp_ctrl_ev_descrips;      /**< List of controlling EV-descriptors, length n_max_evs */

    sk_unint         *p_ev_map;                 /**< Array for mapping EV numbers to remove double EVs after node collapsing,
                                                     length n_max_evs. Only for internal use in the model library. */
} SK_TOPOLOGY;


/*----------------------------- Instance data class --------------------------*/

struct SK_MODEL_DATA;                                 /* forward declaration  */

/**
 * This abstract instance data class contains the interface and generic data for a
 * SiMKit device instance. An SK_INSTANCE_DATA object corresponds with an actual
 * instance in a netlist.
 * This class provides the generic interface to a device instance for the outside
 * world. It provides topology information and methods to get and set instance
 * parameters and perform evaluations of the model equations.
 * The topology of an instance can be in two states, frozen or released.
 * Only the device equations of an instance with a frozen topology can be
 * evaluated. The results of evaluating with a released topology are undefined.
 * Each SiMKit device derives its own  SK_INSTANCE_DATA_DEVICE class from this
 * base class to store device specific data and methods.
 */
typedef struct SK_INSTANCE_DATA
{
    const struct SK_DEVICE_DESCRIPTOR   *p_device_descriptor;       /**< Device descriptor related with this instance */
    const char                          *p_name;                    /**< Name of the instance. */

    const SK_TOPOLOGY                   *p_topology;                /**< Device topology connected with this instance. */

    /**
     * p_set_inst_params sets a specified list of instance parameters to specified
     * values.
     * @param p_inst_data       The instance to set the instance parameters for
     * @param p_param_id_list   Array of parameter IDs to set
     * @param p_value_list      New values of the parameters
     * @param n_params          Number of parameters in p_param_id_list and p_value_list
     * @param p_is_param_changed  Is set to TRUE when at least one parameter value
     *                          has really changed. FALSE otherwise.
     *                          Can be set to NULL if not used.
     * @return  SK_ERR_NONE     All new parameter values have been set.
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is
     *                          invalid. Part of the parameter values may have been set.
     *          SK_ERR_PARAMETER_NOT_WRITEABLE  At least one of the parameters
     *                          is a read-only parameter. All parameter values up to
     *                          this read-only parameter have been set.
     */
    SK_ERROR (*p_set_inst_params)   (       struct SK_INSTANCE_DATA *p_inst_data,
                                      const sk_unint                *p_param_id_list,
                                      const sk_real                 *p_value_list,
                                      const sk_unint                 n_params,
                                            sk_boolean              *p_is_param_changed );

    /**
     * p_get_inst_params gets the values for a list of specified parameters and
     * stores the values in a provided array.
     * @param p_inst_data       The instance to get the instance parameter values for
     * @param p_param_id_list   Array of IDs of the parameters to get the values for
     * @param p_value_list      Array of real to receive the parameter values in. The
     *                          array must be able to hold at least n_params values.
     * @param n_params          Number of parameters to get the value for.
     *                          Minimal size of p_param_id_list and p_value_list.
     * @param p_sim_data        Simulation data. This data might influence the
     *                          parameter values. Can be set to NULL if not used.
     * @return  SK_ERR_NONE     All parameter values have been returned in p_value_list
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is invalid
     *                          Parameter values up to this invalid ID have been stored in
     *                          p_value_list.
     */
    SK_ERROR (*p_get_inst_params)   ( const struct SK_INSTANCE_DATA *p_inst_data,
                                      const sk_unint                *p_param_id_list,
                                            sk_real                 *p_value_list,
                                      const sk_unint                 n_params,
                                      const SK_SIM_DATA             *p_sim_data  );

    /**
     * p_reset_inst_params resets a specified list of instance parameters to their
     * default values.
     * @param p_inst_data       The instance to set the instance parameters for
     * @param model_type        One of the SK_MT_xxx defines to set the type
     * @param p_param_id_list   Array of parameter IDs to set
     * @param n_params          Number of parameters in p_param_id_list and p_value_list
     * @return  SK_ERR_NONE     All new parameter values have been set
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is
     *                          invalid. None of the parameter values has been set.
     */
    SK_ERROR (*p_reset_inst_params)   ( struct SK_INSTANCE_DATA *p_inst_data,
                                  const SK_MODEL_TYPE            model_type,
                                  const sk_unint                *p_param_id_list,
                                  const sk_unint                 n_params  );

    /**
     * p_set_n_terminals sets the number of terminals for an instance.
     * The number of terminals of an instance cannot be changed after it has
     * been set. When this function is called more than once for the same
     * instance, it must be called with the same number of terminals.
     * @param p_inst_data   The instance to set the number of terminals for
     * @param n_terminals   The number of terminals to set
     * @return  SK_ERR_NONE     The number of terminals has been set
     *          SK_ERR_INVALID_NUMBER_OF_TERMINALS
     *                          The specified number of terminals is not supported
     *                          by the device.
     *          SK_ERR_CHANGE_TERMINALS_NOT_ALLOWED
     *                          The function is called more than once with a different
     *                          number of terminals. The new value is not accepted.
     */
    SK_ERROR (*p_set_n_terminals)   (       struct SK_INSTANCE_DATA *p_inst_data,
                                      const sk_unint                 n_terminals);

    /**
     * p_freeze_topology freezes the topology of an instance.
     * Based on the set of model and instance parameter values and the number of
     * terminals this function will set up the lists of terminals, internal
     * nodes, branches and EVs for the instance.
     * @param p_inst_data   The instance object to freeze the topology for
     * @param p_model_data  The model data
     * @return  SK_ERR_NONE The topology has been frozen
     */
    SK_ERROR (*p_freeze_topology)     ( struct SK_INSTANCE_DATA *p_inst_data,
                                  const struct SK_MODEL_DATA    *p_model_data);

    /**
     * p_release_topology releases a previously frozen topology.
     * When a topology is frozen model and instance parameters that influence
     * the topology can not be changed when they would change the topology.
     * To be able to change these parameters it is necessary to release the
     * topology before setting the new parameter values.
     * @param p_inst_data   The instance to release the topology for
     * @return  SK_ERR_NONE The topology has been release
     */
    SK_ERROR (*p_release_topology)  (       struct SK_INSTANCE_DATA *p_inst_data);

    /**
     * p_is_topology_frozen returns TRUE when the topology of the instance
     * has been frozen, FALSE otherwise
     * @param p_inst_data   The instance object to get the topology status for
     * @return  TRUE    The topolgy of the instance is frozen
     *          FALSE   The topology of the instance is NOT frozen
     */
    sk_boolean (*p_is_topology_frozen)(const struct SK_INSTANCE_DATA *p_inst_data);

    /**
     * p_get_topology_id enables the user of the interface (for example an adapter)
     * to test if two instances have the same topology by comparing their topology
     * identifiers. This can be useful to improve the efficiency.
     * @param p_inst_data   The instance to get the topology identifier for
     * @return  The topology identifier (ID number) of the instance
     *          SK_TOPOLOGY_FIXED is returned for fixed topology devices
     */
    sk_unint (*p_get_topology_id)   ( const struct SK_INSTANCE_DATA *p_inst_data);

    /**
     * p_clip_and_scale_params performs clipping and scaling of the model and instance
     * parameters.
     * After clipping, but before scaling and if the topology is frozen, this function will
     * check if the topology would change with the current parameter settings. If this is
     * the case the function returns with an error.
     * @param p_inst_data   The instance to execute the clipping and scaling rules on
     * @param p_model_data  The model data
     * @param p_sim_data    Simulation data. This data might influence the scaling of the
     *                      parameter values. Can be set to NULL if not used.
     * @param temperature   Simulation temperature (Celcius).
     * @param flag          Combination of the SK_IC_xxx flags to control scaling.
     * @return  SK_ERR_NONE Clipping and scaling performed ok
     *          SK_ERR_PARAMS_CHANGE_TOPOLOGY
     *                      The current set of parameters would change a frozen topology.
     */
    SK_ERROR (*p_clip_and_scale_params) ( struct SK_INSTANCE_DATA   *p_inst_data,
                                    const struct SK_MODEL_DATA      *p_model_data,
                                    const SK_SIM_DATA               *p_sim_data,
                                    const sk_real                    temperature,
                                    const SK_INIT_CONTROL            flag);

    /**
     * p_eval_model evaluates the device equations using the electrical parameters,
     * the instance topology, the simuation data and flags and the input EV values and returns
     * the evaluation results in the output EVs and their derivatives.
     * The function also calculates the operating point values and the bias dependent noise parts
     * if requested. The bias dependent info may contain some extra information that depends
     * on the bias condition (for example region information).
     * @param p_inst_data       The instance to evaluate
     * @param p_model_data      The model data
     * @param p_sim_data        Simulation data. Can be set to NULL if not used.
     * @param temperature       Simulation temperature (Celcius)
     * @param flag              Combination of the SK_EC_xxx flags to control the evaluation
     * @param p_input_evs       Input EV values
     * @param p_output_evs      Output EV values and derivatives (if not used, set NULL)
     * @param p_opo_values      Operating point values (if not used, set NULL)
     * @param p_bias_dep_noise_parts  bias dependent parts of noise (if not used, set NULL)
     * @param p_bias_dep_info   bias dependent info  (if not used, set NULL)
     * @return  SK_ERR_NONE Evaluation ok
     */
    SK_ERROR (*p_eval_model)    ( const struct SK_INSTANCE_DATA *p_inst_data,
                                  const struct SK_MODEL_DATA    *p_model_data,
                                  const SK_SIM_DATA             *p_sim_data,
                                  const sk_real                  temperature,
                                  const SK_EVAL_CONTROL          flag,
                                  const SK_INPUT_EV             *p_input_evs,
                                        SK_OUTPUT_EV            *p_output_evs,
                                        sk_real                 *p_opo_values,
                                        sk_real                 *p_bias_dep_noise_parts,
                                        SK_BIAS_DEP_INFO        *p_bias_dep_info);

    /**
     * p_eval_u_noise calculates the (complex) frequency dependent noise part
     * for a noise source, given the frequency and the (complex) bias dependent noise part.
     * @param p_inst_data       The instance to evaluate the noise equations for
     * @param p_model_data      The model data
     * @param noise_src_number  Number of the noise source to get the noise for
     * @param frequency         The frequency to calculate the dependent noise with
     * @param p_h_noise         Structure to receive the h noise (bias dependent noise part)
     * @param p_u_noise         Structure to receive the u noise (frequency dependent noise part)
     * @return  SK_ERR_NONE     Evaluation ok
     */
    SK_ERROR (*p_eval_u_noise)  ( const struct SK_INSTANCE_DATA *p_inst_data,
                                  const struct SK_MODEL_DATA    *p_model_data,
                                  const sk_unint                 noise_src_number,
                                  const sk_real                  frequency,
                                        sk_complex              *p_h_noise,
                                        sk_complex              *p_u_noise);

} SK_INSTANCE_DATA;


/*----------------------------- Model data class -----------------------------*/

/**
 * The model data class contains the interface and generic data for a SiMKit
 * device model. An SK_MODEL_DATA object is an abstract concept to group
 * instances of a device together that have the same model parameter values.
 * Variations in model parameter values are effective for all instances within
 * the group.
 * Each SiMKit model derives its own SK_MODEL_DATA_DEVICE class from this base
 * class to store device specific data and methods.
 */
typedef struct SK_MODEL_DATA
{
    const struct SK_DEVICE_DESCRIPTOR  *p_device_descriptor;    /**< Device descriptor related with this model. */
    const char                         *p_name;                 /**< Name of the model. */
                 SK_LIMITING           *p_limit_info;           /**< Functional interface to implement limiting. */
          sk_real                       Imax;                   /**< Current above which junction currents should be linearised. Set by the adapter */

    /**
     * This function sets the model gender type (P- or N-type) and sets the
     * parameter values to their type dependent default (P- or N-type) values.
     * The model gender type is one of the device types defined by SK_MT_xxx.
     * It depends on the device, which types are valid.
     * @param p_model_data  The model to set the type and default for
     * @param model_type    One of the SK_MT_xxx defines to set the type
     * @return  SK_ERR_NONE                 Type and defaults have been set
     *          SK_ERR_UNKNOWN_TYPE_STRING  Invalid type specified for the model
     */
    SK_ERROR (*p_set_defaults_and_model_type)( struct SK_MODEL_DATA *p_model_data,
                                               const  SK_MODEL_TYPE  model_type );

    /**
     * This function retrieves the model gender type (P- or N-type). The model type is
     * one of the device types defined by SK_MT_xxx. It depends on the device, which
     * types are valid.
     * @param p_model_data  The model to set the type and default for
     * @param model_type    One of the SK_MT_xxx defines to set the type
     * @return  SK_ERR_NONE Type and defaults have been set
     */
    SK_ERROR (*p_get_model_type)( const struct SK_MODEL_DATA    *p_model_data,
                                        SK_MODEL_TYPE           *p_model_type );

    /**
     * This function sets a specified list of model parameters to specified
     * values.
     * @param p_model_data      The model to set the model parameters for
     * @param p_param_id_list   Array of parameter IDs to set
     * @param p_value_list      New values of the parameters
     * @param n_params          Number of parameters in p_param_id_list and p_value_list
     * @param p_is_param_changed  Is set to TRUE when at least one parameter value
     *                          has really changed. FALSE otherwise.
     *                          Can be set to NULL if not used.
     * @return  SK_ERR_NONE     All new parameter values have been set
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is
     *                          invalid. Part of the parameter values may have been set.
     *          SK_ERR_PARAMETER_NOT_WRITEABLE  At least one of the parameters
     *                          is a read-only parameter. All parameter values up to
     *                          this read-only parameter have been set.
     */
    SK_ERROR (*p_set_model_params)  (       struct SK_MODEL_DATA    *p_model_data,
                                      const sk_unint                *p_param_id_list,
                                      const sk_real                 *p_value_list,
                                      const sk_unint                 n_params,
                                            sk_boolean              *p_is_param_changed );

    /**
     * This function gets the values for a list of specified parameters and
     * stores the values in a provided array.
     * @param p_model_data      The model to get the model parameter values for
     * @param p_param_id_list   Array of IDs of the parameters to get the values for
     * @param p_value_list      Array of real to receive the parameter values in. The
     *                          array must be able to hold at least n_params values.
     * @param n_params          Number of parameters to get the value for.
     *                          Minimal size of p_param_id_list and p_value_list.
     * @return  SK_ERR_NONE     All parameter values have been returned in p_value_list
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is invalid
     *                          Parameter values up to this invalid ID have been stored in
     *                          p_value_list.
     */
    SK_ERROR (*p_get_model_params)  ( const struct SK_MODEL_DATA    *p_model_data,
                                      const sk_unint                *p_param_id_list,
                                            sk_real                 *p_value_list,
                                      const sk_unint                 n_params );

    /**
     * This function resets a specified list of model parameters to their
     * default values.
     * @param p_model_data      The model to set the model parameters for
     * @param p_param_id_list   Array of parameter IDs to set
     * @param n_params          Number of parameters in p_param_id_list and p_value_list
     * @return  SK_ERR_NONE     All new parameter values have been set
     *          SK_ERR_UNKNOWN_PARAMETER    At least one of the parameter IDs is
     *                          invalid. None of the parameter values has been set.
     */
    SK_ERROR (*p_reset_model_params)(       struct SK_MODEL_DATA    *p_model_data,
                                      const sk_unint                *p_param_id_list,
                                      const sk_unint                 n_params  );

    /**
     * This function initializes an SK_INSTANCE_DATA object.
     * In reality the function initializes a device specific SK_INSTANCE_DATA_DEVICE object.
     * @param p_inst_data   The instance to initialize
     * @param p_model_data  The model object to create the instance for
     * @param p_inst_name   Name of the instance
     * @param pp_sit        Array of full instance path names (only used by Pstar)
     * @return  SK_ERR_NONE     Initialization ok
     */
    SK_ERROR (*p_init_instance_data)      ( struct SK_INSTANCE_DATA *p_inst_data,
                                            struct SK_MODEL_DATA    *p_model_data,
                                      const char                    *p_inst_name,
                                      const char                   **pp_sit);

} SK_MODEL_DATA;


/*--------------------------- Device descriptor class ------------------------*/

/**
 * The device descriptor class contains the static description of a SiMKit device.
 * The information in this class is valid for all models and instances of this device.
 */
typedef struct SK_DEVICE_DESCRIPTOR
{
    const char                     *p_name;                 /**< Short name of the device. For example: mos1101 */
    const char                     *p_title;                /**< One line description of the device. For example: Compact MOS-Transistor Distortion Model */
    const SK_DEVICE_FAMILY          device_family;          /**< Device family: MOST, PSP, Bipolar or generic */

          sk_unint                  n_model_params;         /**< Number of model parameters. The values of the parameters for each model will be stored in the SK_MODEL_DATA objects. */
          sk_unint                  n_inst_params;          /**< Number of instance parameters. The values of the parameters for each instance will be stored in SK_INSTANCE_DATA objects. */
          sk_unint                  n_op_info;              /**< Number of Operating Point parameters. */
    const SK_PARAM_DESCRIPTOR     **pp_params;              /**< Array containing descriptions (not values!) of all model, instance and operating point parameters.
                                                                 This array contains n_model_params model params followed by n_inst_params
                                                                 instance parameters followed by n_op_info op point parameters. */

    const sk_unint                  n_max_terminals;        /**< Maximum number of terminal nodes this device supports */
    const sk_unint                  n_min_terminals;        /**< Minimum number of terminal nodes this device supports */
    const SK_NODE          * const *pp_all_term_nodes;      /**< List of all possible terminals nodes */

    const sk_unint                  n_max_int_nodes;        /**< Maximum number of internal (= non terminal) nodes this device supports */
    const SK_NODE          * const *pp_all_int_nodes;       /**< List of all possible internal nodes */

    const sk_unint                  n_max_evs;              /**< Maximum number of EVs this device supports */
    const SK_EV_DESCRIP    * const *pp_all_ev_descrip;      /**< List of all possible input EV descriptors */

    const sk_unint                  n_max_branches;         /**< the maximum number of branches supported by this device */
    const sk_unint                  n_max_noise_sources;    /**< the maximum number of noise sources supported by this device */

    const sk_integer                sk_model_data_size;     /**< size of the internal model data (SK_MODEL_DATA_DEVICE) */
    const sk_integer                sk_instance_data_size;  /**< size of the internal instance data (SK_INSTANCE_DATA_DEVICE) */

    const sk_boolean               *p_ev_ig_available;      /**< List of boolean values, specifying for which region (Spectre) or state (Pstar) an initial guess is available for the Electrical Variables (EVs) */ /* Used by Spectre */

    /**
     * p_get_device_info returns static information on the device. The information returned consists
     * of a number of bit flags that are switched on or off. Combinations of flags are or-ed.
     * @return  SK_DEVICE_INFO device information bit flags
     */
    SK_DEVICE_INFO (*p_get_device_info)(void);

    /**
     * This function initializes an SK_MODEL_DATA object and sets all model parameter
     * values to their (type depedendent) default values.
     * In reality the function initializes a device specific SK_MODEL_DATA_DEVICE object.
     * @param p_model_data      The model data to be initialized
     * @param p_model_name      Name of the model
     * @param model_type        Type of the model
     * @return  SK_ERR_NONE     Initialization ok
     */
    SK_ERROR (*p_init_model_data) ( SK_MODEL_DATA      *p_model_data,
                              const char               *p_model_name,
                              const SK_MODEL_TYPE       model_type);

    /**
     * This function initializes the description of a device, such as the generation of the
     * list of all possible topologies for that device. Note that this function may be NULL
     * if a device description doesn't need initialization (this should be checked before use).
     * It is recommended to use p_init_modellib instead.
     */
    SK_ERROR (*p_init_device)(void);

} SK_DEVICE_DESCRIPTOR;


/*----------------------- Model library descriptor class ---------------------*/

/**
 * The model library descriptor class is the entry point for the model library
 * of the SiMKit. It contains functions to get information about the library
 * version and its interface and functions to get access to the device models
 * contained by the model library.
 * The name of the exported symbol for the model library descriptor is:
 * "SK_modellib_descriptor".
 */
typedef struct SK_MODELLIB_DESCRIPTOR
{
    /* NOTE:
     * p_is_interface_compatible shall be the first function of the descriptor! */

    /**
     * Function that can be used by an adapter to check if the interface of the
     * loaded model library is compatible with the interface of the adapter.
     * Even when the interface number of the model library is higher than the interface
     * number of the adapter, the (newer) SiMKit models may still be compatible.
     * @param SK_INTERFACE_VERSION defined in the adapter
     * @return TRUE if the model library interface is compatible, otherwise FALSE.
     */
    sk_boolean (*p_is_interface_compatible)(const int sk_interface_version);

    /**
     * Returns the version number of the model library interface
     * @return Version number
     */
    sk_integer (*p_get_interface_version)(void);

    /**
     * Returns the version string of the SiMKit
     * @return Version string
     */
    const char* (*p_get_simkit_version_string)(void);

    /**
     * Sets the address of the SK_report_to_simulator() function
     * @param aFunc The address of the report function
     * @return none
     */
    void (*p_set_report_to_simulator)(SK_REPORT_TO_SIMULATOR_FUNC aFunc);

    /**
     * Initialize all devices in the modellibrary by calling the p_init_device
     * of all devices that need initialization.
     */
    void (*p_init_modellib)(void);

    /**
     * Returns the number of device descriptors in the model library
     * @return The number of device descriptors
     */
    sk_integer (*p_get_n_device_descriptors)(void);

    /**
     * Gets the pointer to the device_descriptor of a device
     * @param  device_descriptor_index   The index of the device_descriptor (0,...,n_device_descriptors-1)
     * @return The device_descriptor belonging to device_descriptor_index
     */
    SK_DEVICE_DESCRIPTOR* (*p_get_device_descriptor)(const int device_descriptor_index);

} SK_MODELLIB_DESCRIPTOR;


/* Necessary for the MS-Windows version of the SiMKit for ADS */
#ifdef _WIN32
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define finite _finite
#define isnan  _isnan
#define log1p(a) LogOnePlus(a)
#endif

#ifndef _WIN32
#ifdef __cplusplus
}
#endif
#endif

#endif /* _SK_H */

