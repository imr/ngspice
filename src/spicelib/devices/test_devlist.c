#include <stdio.h>
#include <stdlib.h>

#include <ifsim.h>
#include <devdefs.h>

#include "devlist.h"


SPICEdev dummy = {
    {
        "Dummy",
        "A dummy element",

        NULL,
        NULL,
        NULL,

        NULL,
        NULL,

        NULL,
        NULL,
	0
    },

    NULL,
};


SPICEdev *DEVices[] = {
    &dummy,
    &dummy,
    &dummy,
    &dummy
};


int
num_devices(void)
{
    return sizeof(DEVices)/sizeof(SPICEdev *);
}

SPICEdev **
devices(void)
{
    return DEVices;
}


int
main(void)
{
    SPICEdev **dev;
    int count = 0;
    int ret;

    for (dev = first_device(); dev != NULL; dev = next_device(dev)) {
	printf("count: %d\n", count);
	count++;
    }

    if (count == num_devices() + 1) {
	printf("PASSED");
	ret = EXIT_SUCCESS;
    } else {
	printf("FAILED");
	ret = EXIT_FAILURE;
    }
    printf(": test_dev\n");
    return ret;
}
