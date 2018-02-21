// (compile "git grep -l -e 'SPICEdev' -- '*.c' | xargs -P8 -n1 spatch --no-show-diff --smpl-spacing --include src/include/ngspice/devdefs.h --include src/include/ngspice/ifsim.h --sp-file rewrite-SPICEdev.cocci --in-place -D step1 && emacs -q --no-site-file --batch --load rewrite-SPICEdev.el && git commit -am 'rewrite-SPICEdev, coccinelle semantic patch #1'")
//
// (compile "git grep -l -e 'SPICEdev' -- '*.c' | xargs -P8 -n1 spatch --no-show-diff --smpl-spacing --include src/include/ngspice/devdefs.h --include src/include/ngspice/ifsim.h --sp-file rewrite-SPICEdev.cocci --in-place -D step2 && git commit -am 'rewrite-SPICEdev, coccinelle semantic patch #2'")
//
// (compile "git grep -l -e 'SPICEdev' -- '*.c' | xargs -P8 -n1 spatch --smpl-spacing --include src/include/ngspice/devdefs.h --include src/include/ngspice/ifsim.h --sp-file rewrite-SPICEdev.cocci --in-place")
// (compile "emacs -q --no-site-file --batch --load rewrite-SPICEdev.el")
//
// (compile "spatch --sp-file rewrite-SPICEdev.cocci --indent 4 --smpl-spacing --include src/include/ngspice/devdefs.h --include src/include/ngspice/ifsim.h src/spicelib/devices/cap/capinit.c -D step2")
//
// (compile "spatch --longhelp")

virtual step1, step2

@r1 depends on step1@
field list[nilla] fields;
type T;
identifier f;
@@
  typedef struct SPICEdev {
    fields
    T f;
    ...
  } SPICEdev;


@r11 depends on step1@
field list[nilla] fields;
type T;
identifier f;
@@
  struct IFdevice {
    fields
    T f;
    ...
  };

//@r2@
//typedef SPICEdev;
//identifier devinit;
//initializer list[r11.nilla] initialisers;
//initializer i;
//identifier r11.f;
//@@
//  SPICEdev devinit = {
//       {
//       initialisers,
//-     i
//+    .f  =  i
//      , ...
//       }
//    , ...
//  };

@r22 depends on step1@
typedef SPICEdev;
identifier devinit;
initializer list[r1.nilla] initialisers;
initializer i;
identifier r1.f;
@@
  SPICEdev devinit = {
    initialisers,
-   i,
+   .f = i,
    ...
  };

@r23 depends on step1@
typedef SPICEdev;
identifier devinit;
initializer list[r11.nilla] initialisers;
initializer i;
identifier r11.f;
@@
  SPICEdev devinit = {
    .DEVpublic = {
       initialisers,
-     i,
+.f = i,
      ...
      },
    ...
  };

@mop depends on step2@
typedef SPICEdev;
identifier devinit;
expression a,b,c,d,e;
@@
  SPICEdev devinit = {
      .DEVpublic = {
          ... ,
          .modelParms = a,
+         .flags = b,
+
          ...,
-         .flags = b,
      },
+
      ... ,
      .DEVsoaCheck = c,
+     .DEVinstSize = d,
+     .DEVmodSize = e,
+
      ... ,
-     .DEVinstSize = d,
-     .DEVmodSize = e,
  };
