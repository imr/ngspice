/* Dummy main() for Verilator. */

#include "ngspice/cmtypes.h" // For Digital_t
#include "ngspice/cosim.h"   // For struct co_info and prototypes

int main(int argc, char** argv, char**) {
    struct co_info info = {};

    Cosim_setup(&info);
    for (;;)
      (*info.step)(&info);
    return 0;
}
