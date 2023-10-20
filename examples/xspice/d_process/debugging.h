#ifndef INCLUDED_DEBUGGING_H
#define INCLUDED_DEBUGGING_H

static int known_bp(int iargc)
{
    return iargc;
}

void debug_info(int argc, char **argv)
{
#if defined(_MSC_VER) || defined(__MINGW64__)
    fprintf(stderr, "%s pid %d\n", argv[0], _getpid());
#else
    fprintf(stderr, "%s pid %d\n", argv[0], getpid());
#endif

#if !defined(_MSC_VER) && !defined(__MINGW64__)
    if (getenv("GO_TO_SLEEP")) {
        sleep(40);
    }
#endif
#if defined(__MINGW64__)
    if (getenv("GO_TO_SLEEP")) {
        sleep(40);
    }
#endif
#if defined(_MSC_VER)
    if (getenv("GO_TO_SLEEP")) {
        Sleep(60000);
    }
#endif

    (void)known_bp(argc);

    for (int i=0; i<argc; i++) {
        fprintf(stderr, "[%d] %s\n", i, argv[i]);
    }
}
#endif

