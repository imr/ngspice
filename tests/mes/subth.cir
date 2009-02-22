Mesfet level 1 subthreshold characteristics

Vds 1 0 dc 0.1
vids 1 2 dc 0
Vgs 3 0 dc 0

z1 2 3 0 mesmod area=1.4

.model mesmod nmf level=1 rd=46 rs=46 vt0=-1.3
+ lambda=0.03 alpha=3 beta=1.4e-3

.options noacct
.dc vgs -3 0 0.05
.print DC vids#branch


.end
