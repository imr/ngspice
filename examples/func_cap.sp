* func_cap.sp


.func icap_calc(A,B,C,D) '2*A*sqrt(B*C*D)'

.param cap_val = 'max(icap_calc(1,2,3,4))'
VDD 1 0 DC 1
C1 1 0 'cap_val'

.measure tran capacitance param='cap_val'
.measure tran capac2 param='max(icap_calc(1,2,3,4))'

.tran 1ps 100ps

.end
