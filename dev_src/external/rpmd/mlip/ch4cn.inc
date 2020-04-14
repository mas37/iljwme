c  include file for CH4O PES

       common /angles/ theta0(4,4),dtheta0(4,4,4)
       common /bonds/ rcb,rno,rch(4),r0ch,rbh(4),r0hh,r0cb
       common /coords/ tcb(3),tno(3),tch(4,3),tbh(4,3)
       common /delta1/ fdelta(4),hdelta(4)
       common /delta2/ dfdelta(4,4),dhdelta(4,4)
       common /force1/ fk0(4,4),f1(4),dfdc(4,4,4),dfdh(4,4,4)
       common /force2/ fkinf,ak,bk,aa1,aa2,aa3,aa4
       common /fsw1/ a1s,b1s,a2s,b2s,a3s,b3s
       common /ip1/ s1(4),ds1(4),s2(4),ds2(4)
       common /ndx/ nnc,nnb,nnh(4),nno,nc(3),nhb(3),nh(4,3),no(3)
       common /op1/ s3(4),ds3(4)
       common /params/ fch3,hch3
       common /qpdot/ qdot(150),pdot(150)
       common /sphi1/ aphi,bphi,cphi
       common /stheta1/ atheta,btheta,ctheta
       common /stret1/ d1ch,d3ch,a1ch,b1ch,c1ch,
     *                   d1hh,d3hh,ahh,
     *                   d1cb,d3cbi,acb
       common /switch1/ sphi(4),dsphi(4),stheta(4),dstheta(4)
       common /addh2o/ded(3),angh2o(4),fkh2oeq,fkh2o(4),
     *                alph2o,angh2oeq,dkdr(4),dkdx(4,3),
     *                dedo(3),dedhi(3),dedho(3)
       common /add3cb/a3cb,b3cb,rcbsp
