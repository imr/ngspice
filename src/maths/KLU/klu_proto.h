/* function prototypes */

void *KLU_malloc        /* returns pointer to the newly malloc'd block */
(
    /* ---- input ---- */
    size_t n,           /* number of items */
    size_t size,        /* size of each item */
    /* --------------- */
    KLU_common *Common
);

void *KLU_free          /* always returns NULL */
(
    /* ---- in/out --- */
    void *p,            /* block of memory to free */
    /* ---- input --- */
    size_t n,           /* size of block to free, in # of items */
    size_t size,        /* size of each item */
    /* --------------- */
    KLU_common *Common
);

void *KLU_realloc       /* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,        /* requested # of items in reallocated block */
    size_t nold,        /* old # of items */
    size_t size,        /* size of each item */
    /* ---- in/out --- */
    void *p,            /* block of memory to realloc */
    /* --------------- */
    KLU_common *Common
);

Int KLU_free_symbolic
(
    KLU_symbolic **SymbolicHandle,
    KLU_common   *Common
);

KLU_symbolic *KLU_analyze_given     /* returns NULL if error, or a valid
                                       KLU_symbolic object if successful */
(
    /* inputs, not modified */
    Int n,              /* A is n-by-n */
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    Int Puser [ ],      /* size n, user's row permutation (may be NULL) */
    Int Quser [ ],      /* size n, user's column permutation (may be NULL) */
    /* -------------------- */
    KLU_common *Common
);


