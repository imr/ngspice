/* OSDI 0.3 modules for the OSDI tests, written without OpenVAF.
 *
 * Every module is a conductance between terminal 0 and its paired terminal,
 *     g * type * area * m * CONDUCTANCE_FACTOR, or m/r if r is given,
 * so the current of a test shows which module a card selected. Each other
 * terminal has a conductance of 1 mS to ground, so it is never floating.
 */

#include <stddef.h>
#include "osdilegacy.h"

#ifndef CONDUCTANCE_FACTOR
#define CONDUCTANCE_FACTOR 1
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

#define MAX_TERMINALS 5
#define MAX_JACOBIAN_ENTRIES (4 + MAX_TERMINALS - 2)
#define TERMINAL_CONDUCTANCE 1e-3

/* OpenVAF gives every module the instance parameter $mfactor, which ngspice
   calls m, so these modules have it too. l and w only select a bin. */
enum { PARAMETER_R, PARAMETER_AREA, PARAMETER_L, PARAMETER_W, PARAMETER_MFACTOR, INSTANCE_PARAMETER_COUNT };
enum { PARAMETER_TYPE = INSTANCE_PARAMETER_COUNT, PARAMETER_G, PARAMETER_COUNT };

/* An instance parameter, or its default on the model card */
typedef struct {
    double values[INSTANCE_PARAMETER_COUNT];
    bool given[INSTANCE_PARAMETER_COUNT];
} InstanceParameters;

typedef struct {
    InstanceParameters defaults;
    int type;
    bool type_given;
    double g;
    bool g_given;
    uint32_t terminal_count;
} Model;

typedef struct {
    double *jacobian[MAX_JACOBIAN_ENTRIES];
    uint32_t node_mapping[MAX_TERMINALS];
    bool collapsed[1];
    InstanceParameters parameters;
    double conductance;
    uint32_t terminal_count;
} Instance;

static void *access_parameter(void *instance_data, void *model_data, uint32_t id, uint32_t flags)
{
    Instance *instance = instance_data;
    Model *model = model_data;
    const bool set = flags & ACCESS_FLAG_SET;
    if (id < INSTANCE_PARAMETER_COUNT) {
        InstanceParameters *parameters = flags & ACCESS_FLAG_INSTANCE ? &instance->parameters : &model->defaults;
        parameters->given[id] |= set;
        return &parameters->values[id];
    }
    if (id == PARAMETER_TYPE) {
        model->type_given |= set;
        return &model->type;
    }
    model->g_given |= set;
    return &model->g;
}

static double instance_parameter(const Instance *instance, const Model *model, uint32_t id)
{
    return instance->parameters.given[id] ? instance->parameters.values[id] : model->defaults.values[id];
}

static void clear_init_info(OsdiInitInfo *result)
{
    *result = (OsdiInitInfo) { 0, 0, NULL };
}

static void setup_model(Model *model, uint32_t terminal_count, double default_g, OsdiInitInfo *result)
{
    model->terminal_count = terminal_count;
    if (!model->g_given)
        model->g = default_g;
    if (!model->type_given)
        model->type = 1;
    if (!model->defaults.given[PARAMETER_AREA])
        model->defaults.values[PARAMETER_AREA] = 1;
    if (!model->defaults.given[PARAMETER_MFACTOR])
        model->defaults.values[PARAMETER_MFACTOR] = 1;
    clear_init_info(result);
}

static void setup_instance(void *handle, void *instance_data, void *model_data, double temperature,
                           uint32_t connected_terminals, OsdiSimParas *simulator_parameters,
                           OsdiInitInfo *result)
{
    (void) handle;
    (void) temperature;
    (void) connected_terminals;
    (void) simulator_parameters;
    Instance *instance = instance_data;
    const Model *model = model_data;
    const bool r_given = instance->parameters.given[PARAMETER_R] || model->defaults.given[PARAMETER_R];
    const double conductance = r_given
        ? 1 / instance_parameter(instance, model, PARAMETER_R)
        : model->g * model->type * instance_parameter(instance, model, PARAMETER_AREA) * CONDUCTANCE_FACTOR;
    instance->terminal_count = model->terminal_count;
    instance->conductance = instance_parameter(instance, model, PARAMETER_MFACTOR) * conductance;
    clear_init_info(result);
}

static uint32_t eval(void *handle, void *instance, const void *model, const OsdiSimInfo *info)
{
    (void) handle;
    (void) instance;
    (void) model;
    (void) info;
    return 0;
}

static void load_jacobian_resist(void *instance_data, void *model)
{
    (void) model;
    Instance *instance = instance_data;
    *instance->jacobian[0] += instance->conductance;
    *instance->jacobian[1] -= instance->conductance;
    *instance->jacobian[2] -= instance->conductance;
    *instance->jacobian[3] += instance->conductance;
    for (uint32_t i = 4; i < instance->terminal_count + 2; i++)
        *instance->jacobian[i] += TERMINAL_CONDUCTANCE;
}

/* The modules are linear, so the right-hand side has no contribution */
static void load_spice_rhs_dc(void *instance, void *model, double *rhs, double *previous_solution)
{
    (void) instance;
    (void) model;
    (void) rhs;
    (void) previous_solution;
}

#define SETUP_MODEL(module, terminal_count, default_g)                                        \
    static void setup_##module(void *handle, void *model, OsdiSimParas *simulator_parameters, \
                               OsdiInitInfo *result)                                          \
    {                                                                                         \
        (void) handle;                                                                        \
        (void) simulator_parameters;                                                          \
        setup_model(model, terminal_count, default_g, result);                                \
    }

SETUP_MODEL(mos, 4, 1e-3)
SETUP_MODEL(mos_alternate, 4, 2e-3)
SETUP_MODEL(bjt, 5, 3e-3)
SETUP_MODEL(diode, 3, 4e-3)
SETUP_MODEL(resistor, 2, 5e-3)

#define NODE(name) { name, "V", "A", UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, false }
#define ENTRY(node_1, node_2) { { node_1, node_2 }, UINT32_MAX, JACOBIAN_ENTRY_RESIST }

static OsdiNode mos_nodes[] = { NODE("d"), NODE("g"), NODE("s"), NODE("b") };
static OsdiNode bjt_nodes[] = { NODE("c"), NODE("b"), NODE("e"), NODE("s"), NODE("t") };
static OsdiNode diode_nodes[] = { NODE("a"), NODE("c"), NODE("t") };
static OsdiNode resistor_nodes[] = { NODE("p"), NODE("n") };

static OsdiJacobianEntry mos_entries[] = {
    ENTRY(0, 0), ENTRY(0, 2), ENTRY(2, 0), ENTRY(2, 2), ENTRY(1, 1), ENTRY(3, 3)
};
static OsdiJacobianEntry bjt_entries[] = {
    ENTRY(0, 0), ENTRY(0, 2), ENTRY(2, 0), ENTRY(2, 2), ENTRY(1, 1), ENTRY(3, 3), ENTRY(4, 4)
};
static OsdiJacobianEntry diode_entries[] = {
    ENTRY(0, 0), ENTRY(0, 1), ENTRY(1, 0), ENTRY(1, 1), ENTRY(2, 2)
};
static OsdiJacobianEntry resistor_entries[] = {
    ENTRY(0, 0), ENTRY(0, 1), ENTRY(1, 0), ENTRY(1, 1)
};

static char *r_names[] = { "r" };
static char *area_names[] = { "area" };
static char *l_names[] = { "l" };
static char *w_names[] = { "w" };
static char *mfactor_names[] = { "$mfactor" };
static char *type_names[] = { "type" };
static char *g_names[] = { "g" };

static OsdiParamOpvar parameters[] = {
    { r_names, 0, "Resistance", "Ohm", PARA_TY_REAL | PARA_KIND_INST, 0 },
    { area_names, 0, "Area factor", "", PARA_TY_REAL | PARA_KIND_INST, 0 },
    { l_names, 0, "Length", "m", PARA_TY_REAL | PARA_KIND_INST, 0 },
    { w_names, 0, "Width", "m", PARA_TY_REAL | PARA_KIND_INST, 0 },
    { mfactor_names, 0, "Multiplier", "", PARA_TY_REAL | PARA_KIND_INST, 0 },
    { type_names, 0, "Polarity", "", PARA_TY_INT | PARA_KIND_MODEL, 0 },
    { g_names, 0, "Conductance", "S", PARA_TY_REAL | PARA_KIND_MODEL, 0 },
};

#define DESCRIPTOR(module_name, setup, node_list, entry_list)               \
    {                                                                       \
        .name = module_name,                                                \
        .num_nodes = sizeof(node_list) / sizeof(node_list[0]),              \
        .num_terminals = sizeof(node_list) / sizeof(node_list[0]),          \
        .nodes = node_list,                                                 \
        .num_jacobian_entries = sizeof(entry_list) / sizeof(entry_list[0]), \
        .jacobian_entries = entry_list,                                     \
        .collapsed_offset = offsetof(Instance, collapsed),                  \
        .num_params = PARAMETER_COUNT,                                      \
        .num_instance_params = INSTANCE_PARAMETER_COUNT,                    \
        .param_opvar = parameters,                                          \
        .node_mapping_offset = offsetof(Instance, node_mapping),            \
        .jacobian_ptr_resist_offset = offsetof(Instance, jacobian),         \
        .bound_step_offset = UINT32_MAX,                                    \
        .instance_size = sizeof(Instance),                                  \
        .model_size = sizeof(Model),                                        \
        .access = access_parameter,                                         \
        .setup_model = setup,                                               \
        .setup_instance = setup_instance,                                   \
        .eval = eval,                                                       \
        .load_spice_rhs_dc = load_spice_rhs_dc,                             \
        .load_jacobian_resist = load_jacobian_resist,                       \
    }

EXPORT const uint32_t OSDI_VERSION_MAJOR = 0;
EXPORT const uint32_t OSDI_VERSION_MINOR = 3;
EXPORT const OsdiDescriptor03 OSDI_DESCRIPTORS[] = {
    DESCRIPTOR("dummy_mos", setup_mos, mos_nodes, mos_entries),
    DESCRIPTOR("dummy_mos_alternate", setup_mos_alternate, mos_nodes, mos_entries),
    DESCRIPTOR("dummy_bjt", setup_bjt, bjt_nodes, bjt_entries),
    DESCRIPTOR("dummy_diode", setup_diode, diode_nodes, diode_entries),
    DESCRIPTOR("resistor", setup_resistor, resistor_nodes, resistor_entries),
};
EXPORT const uint32_t OSDI_NUM_DESCRIPTORS = sizeof(OSDI_DESCRIPTORS) / sizeof(OSDI_DESCRIPTORS[0]);
