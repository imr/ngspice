Simple coupled transmissionlines
VES IN 0  PULSE (0 1 0N 1.5N 1.5N 4.5N 200N)
R1 IN V1 50
R2 V2 0  10
p1 V1  V2  0 V3  V4 0 cpl1
.model cpl1 cpl
+R = 0.5 0 0.5
+L = 247.3e-9  31.65e-9
+              247.3e-9
+C = 31.4e-12 -2.45e-12
+              31.4e-12
+G = 0 0 0
+length = 0.3048
R3 V3 0 100
R4 V4 0 100
.OPTION NOACCT
.TRAN 0.1N 20N
.PRINT TRAN V(V1) V(V3)
.END
