      subroutine initialize_potential()
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION COORD(21),DX(21)
      dimension dxnum(21)
      character*79 comlin 
C
      COMMON /PDATCM/ D1(3),D3(3),ALPH(3),RE(3),BBETA(3),CC(3),AA(3),    
     *   APARM(5),REFV,TAU,CP,B1,C1
      COMMON /POTXCM/ X(6),Y(6),Z(6),ENERGY,DRDX(15,18),DEDX(18)
C
      include 'ch4cn.inc'
      logical initial
      save initial
      data initial  / .true. /

        INTEGER :: init
        data init /0/
        save init
        call setup(21)
        init = 1

	call mlip_rpmd_init()

      end subroutine initialize_potential

      subroutine get_potential(q,Natoms,Nbeads,V,dVdq,info)
      implicit double precision (a-h,o-z)
        INTEGER :: Natoms, Nbeads, index 
        DOUBLE PRECISION :: q(3,Natoms,Nbeads)
C	Cartesian coordinates of atoms
        DOUBLE PRECISION :: dVdq(3,Natoms,Nbeads), V(Nbeads), Vk
C	dVdq = -force
        INTEGER :: k, info 

      DIMENSION COORD(21),DX(21)
      dimension dxnum(21)
      character*79 comlin 
C
      COMMON /PDATCM/ D1(3),D3(3),ALPH(3),RE(3),BBETA(3),CC(3),AA(3),    
     *   APARM(5),REFV,TAU,CP,B1,C1
      COMMON /POTXCM/ X(6),Y(6),Z(6),ENERGY,DRDX(15,18),DEDX(18)
C
      include 'ch4cn.inc'
      logical initial
         INTEGER :: init

       call setup(21)
       init = 1 

	info = 0  

        v = 0.d0 
        dvdq(:,:,:)  = 0.d0
        DO k = 1, Nbeads
            coord = 0.d0 
            DX = 0.d0 
            Vk = 0.d0 
            call qtocart(q,Natoms,natom,Nbeads,k,coord)
            call SURF(Vk, coord, DX, 21)
            V(k) = Vk           
        index = 0
	do j = 1,7
	do i = 1,3
        index = index + 1 
	dvdq(i,j,K) = dx(index) 
	end do 
	end do         
        END DO

	call MLIP_RPMD_calc(q,Natoms,Nbeads,V,dVdq,info)

      end subroutine get_potential

      subroutine finalize_potential()
	call mlip_rpmd_finalize()

      end subroutine finalize_potential

      subroutine qtocart(q0,nat,natom,nb,inb,cart)      
        implicit none
        DOUBLE PRECISION :: q0(3,nat,nb), cart(21)
        INTEGER :: i,j,k,inb, index 
        INTEGER :: nat, natom,nb 
	cart = 0.d0 
        index = 0 
	do j = 1,nat
	do i= 1,3
	index = index + 1 
        cart(index) = q0(i,j,inb)         
	end do 
	end do 
        return
      end subroutine qtocart






c**************************************************************************

      SUBROUTINE SURF(V, COORD, DX, N3TM)
C
C   System:    CH4+CN based on functional forms of
C              CH4+OH-2008
C              JEG-Marzo-2016
C
C   SETUP must be called once before any calls to SURF.
C   The cartesian coordinates, potential energy, and derivatives of the energy
C   with respect to the cartesian coordinates are passed by the calling  
C   program in the argument list as follows:
C        CALL SURF (V, X, DX, N3TM)
C   where X and DX are arrays dimensioned N3TM, and N3TM must be greater
C   than or equal to 18 (3*number of atoms).  
C   All the information passed to and from the potential energy surface 
C   routine is in hartree atomic units.  
C
C        This potential is written such that:
C                       X(1)  - X(3)  : X, Y, Z for H1
C                       X(4)  - X(6)  : X, Y, Z for C 
C                       X(7)  - X(9)  : X, Y, Z for H3
C                       X(10) - X(12) : X, Y, Z for H4
C                       X(13) - X(15) : X, Y, Z for H2
C                       X(16) - X(18) : X, Y, Z for  C
C                       X(19) - X(21) : X, Y, Z for  N
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      DIMENSION COORD(N3TM),DX(N3TM)
      dimension dxnum(n3tm)
      character*79 comlin 
C
      COMMON /PDATCM/ D1(3),D3(3),ALPH(3),RE(3),BETA(3),CC(3),AA(3),    
     *   APARM(5),REFV,TAU,CP,B1,C1
      COMMON /POTXCM/ X(6),Y(6),Z(6),ENERGY,DRDX(15,18),DEDX(18)
C
      include 'ch4cn.inc'
      logical initial
      save initial
      data initial  / .true. /

C 
C     PUT COORDINATES IN PROPER ARRAYS
C
C Inicialization 
	DO I=1, 150
	  qdot(I)=0.D0
          pdot(I)=0.D0
        ENDDO
c
C Changing to angstroms
C 
      DO 10 I = 1, 21
         qdot(I) = COORD(I)*0.52918d0
 10   CONTINUE
C
c  Reading the parameters for a leps-type pes for the
c  reaction ch4+o -> ch3 + oh from the input file CONST
c
c  The inputed constants are through FORMATTED read statements
c
       open(unit=4,file='CONST',status='old')
c
c  read in indices for the carbon and hydrogen atoms
c
       if (initial) then
       initial=.false.
       write(6,*)
       write(6,80) " <<<<< start of file CONST >>>>>"
       write(6,*)

       read(4,80)comlin
       write(6,80)comlin
       read(4,80)comlin
       write(6,80)comlin

       read(4,80)comlin
       write(6,80)comlin
       read(4,910)nnc,nnb,(nnh(i),i=1,4), nno
       write(6,910)nnc,nnb,(nnh(i),i=1,4), nno

80     format(a79)
910    format(7i4)
c
c  calculate indexes for coordinates
c
       do ind=1,3
         icount=ind-3
         nc(ind)=3*nnc+icount
         nhb(ind)=3*nnb+icount
         do i=1,4
           nh(i,ind)=3*nnh(i)+icount
         no(ind)=3*nno+icount
         enddo
       enddo
c
c  read in parameters for the stretching term
c
       read(4,80)comlin
       write(6,80)comlin
       read(4,930)r0ch,d1ch,d3ch
       write(6,930)r0ch,d1ch,d3ch

       read(4,80)comlin
       write(6,80)comlin
       read(4,930)a1ch,b1ch,c1ch
       write(6,930)a1ch,b1ch,c1ch

       read(4,80)comlin
       write(6,80)comlin
       read(4,930)r0hh,d1hh,d3hh,ahh
       write(6,930)r0hh,d1hh,d3hh,ahh

       read(4,80)comlin
       write(6,80)comlin
       read(4,930)r0cb,d1cb,d3cbi,acb
       write(6,930)r0cb,d1cb,d3cbi,acb

930    format(8f10.5)
c
c  read in parameters for the out of plane bending term
c
       read(4,80)comlin
       write(6,80)comlin
       read(4,930)a3s,b3s,aphi,bphi,cphi,rphi
       write(6,930)a3s,b3s,aphi,bphi,cphi,rphi

       read(4,80)comlin
       read(4,930)atheta,btheta,ctheta
       write(6,80)comlin
       write(6,930)atheta,btheta,ctheta

       read(4,80)comlin
       read(4,930)fch3,hch3
       write(6,80)comlin
       write(6,930)fch3,hch3

c
c  read in parameters for the in plane bending term
c
       read(4,80)comlin
       read(4,930)fkinf,ak,bk,aa1,aa2,aa3,aa4,aa5
       write(6,80)comlin
       write(6,930)fkinf,ak,bk,aa1,aa2,aa3,aa4,aa5
c
c  read in parameters for the h2o bending term
c
       read(4,80) comlin
       read(4,930) fkh2oeq, alph2o, angh2oeq
       write(6,80) comlin
       write(6,930) fkh2oeq, alph2o, angh2oeq
c
c  read in parameters for the D3cb variable term
c
       read(4,80) comlin
       read(4,930) a3cb,b3cb,rcbsp
       write(6,80) comlin
       write(6,930) a3cb,b3cb,rcbsp
c
c  convert to integration units:(Valores en la superficie de Gilbert)
c
c  kcal/mol
c  mdyn A-1        -> 1.0d+05 j/mol...
c  mdyn A rad-1
c
c  NB integration units are:
c
c  energy   in 1.0d+05 j/mol
c  time     in 1.0d-14 s
c  distance in 1.0d-10 m
c  angles   in radians
c  mass     in amu
c
       fact1=0.041840d0
       fact2=6.022045d0
       fact3=2.d0*3.1415926d0/360.d0
c 
       d1ch=d1ch*fact1
       d3ch=d3ch*fact1
       d1cb=d1cb*fact1
       d3cbi=d3cbi*fact1
       a3cb=a3cb*fact1
       d1hh=d1hh*fact1
       d3hh=d3hh*fact1
       fch3=fch3*fact2
       hch3=hch3*fact2
       fkinf=fkinf*fact2
       ak=ak*fact2
       fkh2oeq=fkh2oeq*fact2
       angh2oeq=angh2oeq*fact3

       close(unit=4)
       write(6,*)
       write(6,80) " <<<<< end of file CONST >>>>>"
       write(6,*)
       endif
c
c  calculate relative coordinates and bond lengths
c
       en=0.0d0

       call coorden
c
c  calculate switching functions
c
       call switchf
c
c  calculate reference angles and their derivatives
c
       call refangles
c
c  calculate stretching potential
c
       call stretch(vstr)
c
c  calculate out of plane bending potential
c
       call opbend(vop)
c
c  calculate in plane bending potential
c
       call ipbend(vip)
c
c  nb: total potential energy is vstr+vop+vip
c
       en=vstr+vop+vip
c
c changing from 10(5) j/mol to au
c
       en = en*0.03812D0
       V = en 
c
c  copy the pdots to the
c  appropriate elements in dv. dx en POLYRATE.
c
       do n=1,N3TM  
           DX(n)=0.0d0
       enddo
c
C Reordering and transformation from 10(5)j/mol/A 
C to au
C 
       do ind=1,21
c         DX(ind)=pdot(ind)*0.0201723d0
          DX(ind)=pdot(ind)*0.03812d0*0.52918d0
       enddo
c
c ****************
C     Calculo de derivadas numericas de la energia con
C     respecto a las coordenadas.
c
      goto 999
      PASO= 1.00D-7
      do I = 1,21
        qdot(I) = qdot(I) + PASO
        call coorden
        call switchf
        call refangles
        call stretch(vstr)
        call opbend(vop)
        call ipbend(vip)
        en=vstr+vop+vip
        en = en*0.03812D0
        DXNUM(I) = (en - V)/PASO
        DXNUM(I) = DXNUM(I)*0.52918d0
        qdot(I) = qdot(I) - PASO
      enddo
       DO I=1,21
          ERRDX=ABS((DX(I)-DXNUM(I))/DX(I)*100.d0)
          IF ((ERRDX.GT.1.0d0).AND.(DX(I).GT.1.d-5)) THEN
              WRITE (*,*) "ERROR EN DERIVADAS:",
     *                     I, DX(I), DXNUM(I),ERRDX
              stop
          ENDIF
          WRITE (69,*) DX(I),DXNUM(I),ERRDX
          DX(I)=DXNUM(I)
       ENDDO
C **********************
999    return
       end
c
c******************************************************
c
       subroutine coorden
c
c  calculates relative coordinates and bond lengths
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
c  calculate relative coordinates
c
       do ind=1,3
         tcb(ind)=qdot(nc(ind))-qdot(nhb(ind))
         tno(ind)=qdot(no(ind))-qdot(nhb(ind))
         do i=1,4
           tch(i,ind)=qdot(nc(ind))-qdot(nh(i,ind))
           tbh(i,ind)=qdot(nhb(ind))-qdot(nh(i,ind))
         enddo
       enddo
c
c  calculate bond lengths
c
       rcb=sqrt(tcb(1)*tcb(1)+tcb(2)*tcb(2)+tcb(3)*tcb(3))
       rno=sqrt(tno(1)*tno(1)+tno(2)*tno(2)+tno(3)*tno(3))
       do i=1,4
         rch(i)=sqrt(tch(i,1)*tch(i,1)+tch(i,2)*tch(i,2)+
     *                tch(i,3)*tch(i,3))
         rbh(i)=sqrt(tbh(i,1)*tbh(i,1)+tbh(i,2)*tbh(i,2)+
     *                tbh(i,3)*tbh(i,3))
       enddo
       return
       end
c
c******************************************************
c
c
       subroutine refangles
c
c  subroutine calculates reference angles for the "in-plane" potential
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
       dimension sumd2(4),sumd4(4),ddr(4,4)
       tau=acos(-1.0d0/3.0d0)
       pi=4.0d0*atan(1.0d0)
       halfpi=0.5d0*pi
       twopi=2.0d0*pi
c
c  set diagonal elements to zero
c
       do i=1,4
         theta0(i,i)=0.0d0
         do k=1,4
           dtheta0(i,i,k)=0.0d0
         enddo
       enddo
c
c  calculate reference angles
c
       theta0(1,2)=tau+(tau-halfpi)*(sphi(1)*sphi(2)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(3)*stheta(4)-1.0d0)
       theta0(1,3)=tau+(tau-halfpi)*(sphi(1)*sphi(3)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(2)*stheta(4)-1.0d0)
       theta0(1,4)=tau+(tau-halfpi)*(sphi(1)*sphi(4)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(2)*stheta(3)-1.0d0)
       theta0(2,3)=tau+(tau-halfpi)*(sphi(2)*sphi(3)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(1)*stheta(4)-1.0d0)
       theta0(2,4)=tau+(tau-halfpi)*(sphi(2)*sphi(4)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(1)*stheta(3)-1.0d0)
       theta0(3,4)=tau+(tau-halfpi)*(sphi(3)*sphi(4)-1.0d0)
     *             +(tau-twopi/3.0d0)*(stheta(1)*stheta(2)-1.0d0)
c
c  calculate the derivatives of theta0(i,j) in terms of rch(k)
c  quantity calulated is dtheta0(i,j,k)
c
c  derivatives wrt rch(1)
c
       dtheta0(1,2,1)=(tau-halfpi)*dsphi(1)*sphi(2)
       dtheta0(1,3,1)=(tau-halfpi)*dsphi(1)*sphi(3)
       dtheta0(1,4,1)=(tau-halfpi)*dsphi(1)*sphi(4)
       dtheta0(2,3,1)=(tau-twopi/3.0d0)*dstheta(1)*stheta(4)
       dtheta0(2,4,1)=(tau-twopi/3.0d0)*dstheta(1)*stheta(3)
       dtheta0(3,4,1)=(tau-twopi/3.0d0)*dstheta(1)*stheta(2)
c
c  derivatives wrt rch(2)
c
       dtheta0(1,2,2)=(tau-halfpi)*sphi(1)*dsphi(2)
       dtheta0(1,3,2)=(tau-twopi/3.0d0)*dstheta(2)*stheta(4)
       dtheta0(1,4,2)=(tau-twopi/3.0d0)*dstheta(2)*stheta(3)
       dtheta0(2,3,2)=(tau-halfpi)*dsphi(2)*sphi(3)
       dtheta0(2,4,2)=(tau-halfpi)*dsphi(2)*sphi(4)
       dtheta0(3,4,2)=(tau-twopi/3.0d0)*stheta(1)*dstheta(2)
c
c  derivatives wrt rch(3)
c
       dtheta0(1,2,3)=(tau-twopi/3.0d0)*dstheta(3)*stheta(4)
       dtheta0(1,3,3)=(tau-halfpi)*sphi(1)*dsphi(3)
       dtheta0(1,4,3)=(tau-twopi/3.0d0)*stheta(2)*dstheta(3)
       dtheta0(2,3,3)=(tau-halfpi)*sphi(2)*dsphi(3)
       dtheta0(2,4,3)=(tau-twopi/3.0d0)*stheta(1)*dstheta(3)
       dtheta0(3,4,3)=(tau-halfpi)*dsphi(3)*sphi(4)
c
c  derivatives wrt rch(4)
c
       dtheta0(1,2,4)=(tau-twopi/3.0d0)*stheta(3)*dstheta(4)
       dtheta0(1,3,4)=(tau-twopi/3.0d0)*stheta(2)*dstheta(4)
       dtheta0(1,4,4)=(tau-halfpi)*sphi(1)*dsphi(4)
       dtheta0(2,3,4)=(tau-twopi/3.0d0)*stheta(1)*dstheta(4)
       dtheta0(2,4,4)=(tau-halfpi)*sphi(2)*dsphi(4)
       dtheta0(3,4,4)=(tau-halfpi)*sphi(3)*dsphi(4)
c
c  fill in the other half of the matrix
c
        do i=1,3
          do j=i+1,4
            theta0(j,i)=theta0(i,j)
            do k=1,4
              dtheta0(j,i,k)=dtheta0(i,j,k)
            enddo
          enddo
        enddo
       return
       end
c
c******************************************************
c
c
       subroutine stretch(vstr)
c
c  subroutine to calculate leps-type stretching potential and its
c  derivatives
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
       dimension vqch(4),vjch(4),vqbh(4),vjbh(4),vq(4),vj(4),
     *           achdc(3),achdh(4,3)
c
c  calculate avergage bond length for the methane moiety
c
       rav=(rch(1)+rch(2)+rch(3)+rch(4))/4.0d0
c
c  initialise:
c
       vstr=0.0d0
c
c  calculate d3cb and dd3cb
c
       texp = exp(-(4.d0*(rav-rcbsp)/b3cb)**4.d0) 
       d3cb = (d3cbi-a3cb) + a3cb * texp
       dd3cb = -4.d0*a3cb*texp*(rav-rcbsp)**3.d0*(4.d0/b3cb)**4.d0
c      write (95,*) a3cb,b3cb,rcbsp,rav, d3cbi,d3cb
c
c  ach:
c
c  nb: in double precision tanh(19.0d0)=1.0d0 and we put the if statement
c  in to avoid overflow/underflow errors
c
       arga=c1ch*(rav-r0ch)
       if(arga.lt.19.0d0)then
         ach=a1ch+b1ch*(tanh(arga)+1.0d0)*0.5d0
         dumach=b1ch*c1ch/(2.0d0*cosh(arga)**2)
       else
         ach=a1ch+b1ch
         dumach=0.0d0
       endif
c
c  calculate singlet: e1, triplet: e3 energies and vq and vj
c  terms for each bond
c
       e1=d1cb*(exp(-2.0d0*acb*(rcb-r0cb))-2.0d0*exp(-acb*(rcb-r0cb)))
       e3=d3cb*(exp(-2.0d0*acb*(rcb-r0cb))+2.0d0*exp(-acb*(rcb-r0cb)))
       vqcb=(e1+e3)*0.5d0
       vjcb=(e1-e3)*0.5d0
c
c C-N  new term (simple morse term)
c
       dt=(rno-1.172d0)
       expterm=exp(-0.80d0*dt)
       vno=80.0d0*(1.d0-expterm)**2.d0
c      dt=(rno-r0hh)
c      expterm=exp(-ahh*dt)
c      vno=d1hh*(1.d0-expterm)**2.d0
c
       do i=1,4
         e1=d1ch*(exp(-2.0d0*ach*(rch(i)-r0ch))
     *              -2.0d0*exp(-ach*(rch(i)-r0ch)))
         e3=d3ch*(exp(-2.0d0*ach*(rch(i)-r0ch))
     *              +2.0d0*exp(-ach*(rch(i)-r0ch)))
         vqch(i)=(e1+e3)*0.5d0
         vjch(i)=(e1-e3)*0.5d0
         e1=d1hh*(exp(-2.0d0*ahh*(rbh(i)-r0hh))
     *              -2.0d0*exp(-ahh*(rbh(i)-r0hh)))
         e3=d3hh*(exp(-2.0d0*ahh*(rbh(i)-r0hh))
     *              +2.0d0*exp(-ahh*(rbh(i)-r0hh)))
         vqbh(i)=(e1+e3)*0.5d0
         vjbh(i)=(e1-e3)*0.5d0
c
c  calculate 3 body potential
c
         vq(i)=vqch(i)+vqcb+vqbh(i)
         vj(i)=-sqrt(((vjch(i)-vjcb)**2+(vjcb-vjbh(i))**2
     *                 +(vjbh(i)-vjch(i))**2)*0.5d0)
         vstr=vstr+vq(i)+vj(i)
       enddo
c jcc
       vstr=vstr+vno
c jcc
c
c  partial derivatives
c
c  derivative of OH morse term with respect to dt:
c
c      deddt = 2.0d0 * ahh * d1hh * (1.0d0-expterm) *
c    *         expterm
c
       deddt = 2.0d0 * 0.8d0*80.0d0* (1.0d0-expterm) *
     *         expterm
c  chain rule: derivatives with respect to cartesian coordinates.
c
       de = deddt / rno
       do i=1,3
          ded(i) = de * tno(i)
       enddo
c
c  first we need the derivative of ach:
c
       do ind=1,3
         achdc(ind)=dumach*(tch(1,ind)/rch(1)+tch(2,ind)/rch(2)
     *            +tch(3,ind)/rch(3)+tch(4,ind)/rch(4))/4.0d0
         do i=1,4
           achdh(i,ind)=-dumach*tch(i,ind)/rch(i)/4.0d0
         enddo
       enddo
       dumqcb=-acb*((d1cb+d3cb)*exp(-2.0d0*acb*(rcb-r0cb))-
     *         (d1cb-d3cb)*exp(-acb*(rcb-r0cb)))/rcb
c
c  calculate cartesian derivatives:
c  looping over ch(i) and bh(i)
c
       do i=1,4
         dumqbh=-ahh*((d1hh+d3hh)*exp(-2.0d0*ahh*(rbh(i)-r0hh))-
     *           (d1hh-d3hh)*exp(-ahh*(rbh(i)-r0hh)))/rbh(i)
c
c  addittional term: derv of D3cb wrt r
c
         addd3 = dd3cb*(exp(-2.0d0*acb*(rcb-r0cb))+
     *         2.0d0*exp(-acb*(rcb-r0cb)))
         addd3 = addd3/4.d0 * 0.5d0 / rcb
         dumqbh = dumqbh + addd3
         factj=0.5d0/vj(i)
         dumjcb=-acb*((d1cb-d3cb)*exp(-2.0d0*acb*(rcb-r0cb))
     *            -(d1cb+d3cb)*exp(-acb*(rcb-r0cb)))*factj/rcb
c
         dumjbh=-ahh*((d1hh-d3hh)*exp(-2.0d0*ahh*(rbh(i)-r0hh))
     *            -(d1hh+d3hh)*exp(-ahh*(rbh(i)-r0hh)))*factj/rbh(i)
c
c  addittional term: derv of D3cb wrt r
c
         addd3 = dd3cb*(exp(-2.0d0*acb*(rcb-r0cb))+
     *           2.0d0*exp(-acb*(rcb-r0cb)))
         addd3 = addd3/4.d0 * 0.5d0 * factj / rcb
         dumjbh = dumjbh - addd3
         do ind=1,3
c
c  deriv wrt hb:
c
                  pdot(nhb(ind))=pdot(nhb(ind))
     *             -tcb(ind)*dumqcb+tbh(i,ind)*dumqbh
     *            +(vjch(i)-vjcb)*(dumjcb*tcb(ind))
     *            +(vjcb-vjbh(i))*(-dumjcb*tcb(ind)-dumjbh*tbh(i,ind))
     *            +(vjbh(i)-vjch(i))*dumjbh*tbh(i,ind)
c
c  dvqch(i)/dc
c
           dumqch=-(ach*tch(i,ind)/rch(i)+achdc(ind)*(rch(i)-r0ch))
     *              *((d1ch+d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *                 -(d1ch-d3ch)*exp(-ach*(rch(i)-r0ch)))
               pdot(nc(ind))=pdot(nc(ind))+dumqch+tcb(ind)*dumqcb
c
c  dvqch(i)/dh(i)
c
           dumqhi=(ach*tch(i,ind)/rch(i)-achdh(i,ind)*(rch(i)-r0ch))
     *              *((d1ch+d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *                 -(d1ch-d3ch)*exp(-ach*(rch(i)-r0ch)))
              pdot(nh(i,ind))=pdot(nh(i,ind))+dumqhi-tbh(i,ind)*dumqbh
c
c  dvjch(i)/dc
c
           dumjch=-(ach*tch(i,ind)/rch(i)+achdc(ind)*(rch(i)-r0ch))
     *              *((d1ch-d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *               -(d1ch+d3ch)*exp(-ach*(rch(i)-r0ch)))*factj
c
c  dvj(i)/dnc(ind)
c
           pdot(nc(ind))=pdot(nc(ind))
     *            +(vjch(i)-vjcb)*(dumjch-dumjcb*tcb(ind))
     *            +(vjcb-vjbh(i))*dumjcb*tcb(ind)
     *            -(vjbh(i)-vjch(i))*dumjch
c
c  dvjch(i)/dh(i)
c
           dumjhi=(ach*tch(i,ind)/rch(i)-achdh(i,ind)*(rch(i)-r0ch))
     *              *((d1ch-d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *               -(d1ch+d3ch)*exp(-ach*(rch(i)-r0ch)))*factj
c
c  dvj(i)/dnh(i,ind)
c
            pdot(nh(i,ind))=pdot(nh(i,ind))
     *            +(vjch(i)-vjcb)*dumjhi
     *            +(vjcb-vjbh(i))*dumjbh*tbh(i,ind)
     *            +(vjbh(i)-vjch(i))*(-dumjbh*tbh(i,ind)-dumjhi)
c
c  dv(i)/dh(j)
c
           do k=1,3
             j=i+k
             if(j.gt.4)j=j-4
             dumqhj=-achdh(j,ind)*(rch(i)-r0ch)
     *                 *((d1ch+d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *                    -(d1ch-d3ch)*exp(-ach*(rch(i)-r0ch)))
             dumjhj=-achdh(j,ind)*(rch(i)-r0ch)
     *                 *((d1ch-d3ch)*exp(-2.0d0*ach*(rch(i)-r0ch))
     *                  -(d1ch+d3ch)*exp(-ach*(rch(i)-r0ch)))*factj
             pdot(nh(j,ind))=pdot(nh(j,ind))+dumqhj
     *            +(vjch(i)-vjcb)*dumjhj
     *            -(vjbh(i)-vjch(i))*dumjhj
           enddo
         enddo
       enddo
c
c add derivatives of OH morse term
c
       do ind=1,3
          pdot(nhb(ind))=pdot(nhb(ind)) - ded(ind)
          pdot(no(ind))=pdot(no(ind)) + ded(ind)
       enddo
c
       return
       end
c
c******************************************************
c
c
       subroutine opbend(vop)
c
c  subroutine calculates symmetrized vop potential and derivatives
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
       double precision norma
       dimension sumd2(4),sumd4(4)
       dimension in(3),a(3),b(3),axb(3),c(4,3),argd(4)
c
c
       vop=0.0d0
c
c  calculate force constants and their derivatives
c
       call opforce
c
c  calculate out-of-plane angle and derivatives
c
       do i=1,4
         j=i+1
         if(j.gt.4)j=j-4
         k=j+1
         if(k.gt.4)k=k-4
         l=k+1
         if(l.gt.4)l=l-4
c
c  jcc 6/4/99
c  modification to ensure that the set of methane CH bond vector 
c  (rj,rk,rl) is a right-handed set
c
       in(1)=j
       in(2)=k
       in(3)=l
c
c  vector a is rk-rj, vector b is rl-rj
c
       do ind=1,3
         a(ind)=qdot(nh(k,ind))-qdot(nh(j,ind))
         b(ind)=qdot(nh(l,ind))-qdot(nh(j,ind))
       enddo
c
c  axb is vector a cross b
c
       axb(1)=a(2)*b(3)-a(3)*b(2)
       axb(2)=a(3)*b(1)-a(1)*b(3)
       axb(3)=a(1)*b(2)-a(2)*b(1)
       norma=axb(1)*axb(1)+axb(2)*axb(2)+axb(3)*axb(3)
       norma=sqrt(norma)
c
c  c is position vector of h(ii): calculate c(j),c(k),c(l)
c
       do ii=1,3
         do ind=1,3
           c(in(ii),ind)=-tch(in(ii),ind)/rch(in(ii))
         enddo
       enddo
c
c  argd is the dot product axb dot c
c
       do ii=1,3
         argd(in(ii))=axb(1)*c(in(ii),1)+axb(2)*c(in(ii),2)
     *                                +axb(3)*c(in(ii),3)
         argd(in(ii))=argd(in(ii))/norma
c
c  if argd > 0 we need to switch vectors k and l around
c
         if (argd(in(ii)).gt.0.d0) then
             itemp=k
             k=l
             l=itemp
         endif
       enddo
c
c  jcc 6/4/99
c
c  subroutine performs sum over j, k, l
c  sum2 = sum delta**2
c  sum4 = sum delta**4
c
         call calcdelta(i,j,k,l,sum2,sum4)
         sumd2(i)=sum2
         sumd4(i)=sum4
         vop=vop+fdelta(i)*sumd2(i)+hdelta(i)*sumd4(i)
       enddo
       do i=1,4
         do j=1,4
c
c  overall derivatives of force constants i wrt the bond-length rch(j)
c
           ddr=dfdelta(i,j)*sumd2(i)+dhdelta(i,j)*sumd4(i)
c
c  calculate derivatives in terms of cartesian coordinates:
c
           do ind=1,3
             pdot(nh(j,ind))=pdot(nh(j,ind))-tch(j,ind)*ddr/rch(j)
             pdot(nc(ind))=pdot(nc(ind))+tch(j,ind)*ddr/rch(j)
           enddo
         enddo
       enddo
       return
       end
c
c******************************************************
c
c
       subroutine ipbend(vip)
c
c  subroutine calculates symmetrised in plane bend term
c  and its derivatives
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
       dimension costh(4,4),theta(4,4),dth(4,4)
c
c  initialise
c
       vip=0.0d0
c
c  calculate force constants: fk0(i,j), f1(i)
c  and derivatives wrt rch(k) and rbh(k): dfdc(i,j,k), dfdh(i,j,k)
c
       call ipforce
c
c  calculate theta(i,j) and in plane bend potential
c
       do i=1,3
         do j=i+1,4
           costh(i,j)=tch(i,1)*tch(j,1)+tch(i,2)*tch(j,2)
     *                       +tch(i,3)*tch(j,3)
           costh(i,j)=costh(i,j)/rch(i)/rch(j)
           theta(i,j)=acos(costh(i,j))
           dth(i,j)=theta(i,j)-theta0(i,j)
           vip=vip+0.5d0*fk0(i,j)*f1(i)*f1(j)*dth(i,j)**2
c
c  calculate partial derivatives wrt cartesian coordinates
c
c  calculate pdots wrt theta:
c
           termth=-1.0d0/sqrt(1.0d0-costh(i,j)*costh(i,j))
           do ind=1,3
             dthi=-tch(j,ind)/rch(i)/rch(j)
     *                  +costh(i,j)*tch(i,ind)/rch(i)/rch(i)
             dthi=dthi*termth
             dthj=-tch(i,ind)/rch(i)/rch(j)
     *                  +costh(i,j)*tch(j,ind)/rch(j)/rch(j)
             dthj=dthj*termth
             dthc=-(dthi+dthj)
             pdot(nh(i,ind))=pdot(nh(i,ind))
     *                       +fk0(i,j)*f1(i)*f1(j)*dthi*dth(i,j)
             pdot(nh(j,ind))=pdot(nh(j,ind))
     *                       +fk0(i,j)*f1(i)*f1(j)*dthj*dth(i,j)
             pdot(nc(ind))=pdot(nc(ind))
     *                       +fk0(i,j)*f1(i)*f1(j)*dthc*dth(i,j)
             do k=1,4
c
c  calculate pdots wrt force constants and wrt theta0
c
               dth0k=-dtheta0(i,j,k)*tch(k,ind)/rch(k)
               dth0c=-dth0k
               pdot(nh(k,ind))=pdot(nh(k,ind))
     *                -0.5d0*tch(k,ind)*dfdc(i,j,k)*dth(i,j)**2/rch(k)
     *                -0.5d0*tbh(k,ind)*dfdh(i,j,k)*dth(i,j)**2/rbh(k)
     *                      -fk0(i,j)*f1(i)*f1(j)*dth0k*dth(i,j)
               pdot(nc(ind))=pdot(nc(ind))
     *                +0.5d0*tch(k,ind)*dfdc(i,j,k)*dth(i,j)**2/rch(k)
     *                      -fk0(i,j)*f1(i)*f1(j)*dth0c*dth(i,j)
               pdot(nhb(ind))=pdot(nhb(ind))
     *                +0.5d0*tbh(k,ind)*dfdh(i,j,k)*dth(i,j)**2/rbh(k)
             enddo
           enddo
         enddo
       enddo
c
c  Now, calculate h2o bending energies. 
c
c  First, calculate the angles:
c
       do i = 1,4
         dot = 0.d0
         do j = 1,3
           dot = dot - tno(j)*tbh(i,j)
         enddo
         cosine = dot / (rno*rbh(i))
         cosine = min(1.d0, max(-1.d0, cosine))
         angh2o(i) = acos(cosine)
       enddo
c
c      write (*,*) "angulos:"
c      do i=1,4
c      write (*,*) i, angh2o(i)*360.d0/(2.d0*3.1415926d0)
c      enddo
c
c  Now, calculate each force constant fkh2o(i) as a function of the O-H(I)
c  distance, rbh(i)
c
       do i=1,4
          arga= alph2o * (rbh(i) - r0hh)
          if(arga.lt.19.0d0)then
            fkh2o(i) = fkh2oeq * (1 - tanh(arga))
          else
            fkh2o(i) = 0.0d0
          endif
       enddo
c
c  Now calculate the contribution to the energy for each H(I)-O-H(O)
c  harmonic bending term
c
      do i=1,4
           dang = (angh2o(i) - angh2oeq)
           vip=vip+0.5d0* fkh2o(i) * dang * dang
      enddo
c
c  Calculate the derivatives of the bending hamonic terms
c  (in order to make the code more efficient, some of these terms
c  can be calculated at the same time as the terms for the energy,
c  but in order to make it more clear we will keep them separated
c  even though it means recalculating some stuff)
c
c  d vip                                   d fkh2o(i)    d r
c ------ = 1/2 * (angh2o(i)-angh2oeq)**2 * ---------- * -----  +
c  d x                                       d r         d x
c
c                                            d angh2o(i)
c         + fkh2o(i) * (angh2o(i)-angh2oeq) * -----------
c                                               d x
c
c       
c  First, calculate d fkh2o(i) / d r
c
      do i=1,4
         deno = cosh(alph2o * (rbh(i) - r0hh))
         dkdr(i) = - (fkh2oeq * alph2o) / (deno*deno)
      enddo
c
c  Now we calculate d fhk2o(i) / d x 
c
      do i=1,4
         dkdr(i) = dkdr(i)/rbh(i)
         do j=1,3
            dkdx(i,j) = dkdr(i) * tbh(i,j)
         enddo
      enddo
c
c  We add the contribution of the first half of d vip / d x to pdot
c
      do i=1,4
         do j=1,3
            term1 = 0.5d0 * (angh2o(i)-angh2oeq)**2.d0
            pdot(nhb(j)) = pdot(nhb(j)) + dkdx(i,j) * term1
            pdot(nh(i,j)) = pdot(nh(i,j)) - dkdx(i,j) * term1
         enddo
      enddo
c
c  Now calculate the second half:
c
      do i=1,4
        dstda = fkh2o(i) * (angh2o(i)-angh2oeq)
c
c  contributions of d angle / d x adapted from the tinker code (eangle1.f)
c
        xp = tno(2)*tbh(i,3) - tno(3)*tbh(i,2)
        yp = tno(3)*tbh(i,1) - tno(1)*tbh(i,3)
        zp = tno(1)*tbh(i,2) - tno(2)*tbh(i,1)
        rp = sqrt (xp*xp + yp*yp + zp*zp)
c
c  check of lineality
c
        if (rp.lt.1.d-6) rp=1.d-6
c
c       write (99,*) rp, rno*rbh(i)*sin(angh2o(i))
c
        terma = dstda / (rbh(i)*rbh(i)*rp)
        termc = dstda / (rno*rno*rp)
        dedhi(1) = - terma * (tbh(i,2)*zp-tbh(i,3)*yp)
        dedhi(2) = - terma * (tbh(i,3)*xp-tbh(i,1)*zp)
        dedhi(3) = - terma * (tbh(i,1)*yp-tbh(i,2)*xp)
        dedho(1) = - termc * (tno(2)*zp-tno(3)*yp)
        dedho(2) = - termc * (tno(3)*xp-tno(1)*zp)
        dedho(3) = - termc * (tno(1)*yp-tno(2)*xp)
        dedo(1) = -dedhi(1) - dedho(1)
        dedo(2) = -dedhi(2) - dedho(2)
        dedo(3) = -dedhi(3) - dedho(3)
c
c  add these contributions to pdot
c
c     write (*,*) "Contribuciones segunda parte"
c     write (99,*) rbh(i), angh2o(i)
        do j=1,3
c     write (99,*) i,j, (nhb(j)), dedo(j)
c     write (99,*) i,j, (nh(i,j)), dedhi(j)
c     write (99,*) i,j, (no(j)), dedho(j)
           pdot(nhb(j)) = pdot(nhb(j)) + dedo(j)
           pdot(nh(i,j)) = pdot(nh(i,j)) + dedhi(j)
           pdot(no(j)) = pdot(no(j)) + dedho(j)
        enddo
      enddo
c
       return
       end
c
c*************************************************************************
c
       subroutine calcdelta(i,j,k,l,sum2,sum4)
c
c  subroutine calculates out of plane angle delta, loops
c  through delta(i,j), delta(i,k), delta(i,l)
c
c   also calculates the derivatives wrt delta
c
       implicit double precision (a-h,o-z)
       double precision norma
       include 'ch4cn.inc'
c
       dimension  delta(4),in(3),a(3),b(3),axb(3),c(4,3),argd(4),
     *            daxb(4,3,3),cdot(4,3,3),atemp2(3)
c
c  initialise
c
       sum2=0.0d0
       sum4=0.0d0
c
c  set j,k,l indices
c
       in(1)=j
       in(2)=k
       in(3)=l
c
c  vector a is rk-rj, vector b is rl-rj
c
       do ind=1,3
         a(ind)=qdot(nh(k,ind))-qdot(nh(j,ind))
         b(ind)=qdot(nh(l,ind))-qdot(nh(j,ind))
       enddo
c
c  axb is vector a cross b
c
       axb(1)=a(2)*b(3)-a(3)*b(2)
       axb(2)=a(3)*b(1)-a(1)*b(3)
       axb(3)=a(1)*b(2)-a(2)*b(1)
       norma=axb(1)*axb(1)+axb(2)*axb(2)+axb(3)*axb(3)
       norma=sqrt(norma)
c
c  c is position vector of h(ii): calculate c(j),c(k),c(l)
c
       do ii=1,3
         do ind=1,3
           c(in(ii),ind)=-tch(in(ii),ind)/rch(in(ii))
         enddo
       enddo
c
c  argd is the dot product axb dot c
c
       do ii=1,3
         argd(in(ii))=axb(1)*c(in(ii),1)+axb(2)*c(in(ii),2)
     *                                +axb(3)*c(in(ii),3)
         argd(in(ii))=argd(in(ii))/norma
         delta(in(ii))=acos(argd(in(ii)))-theta0(i,in(ii))
c        write(*,*) 'theta,delta',theta0(i,in(ii)),delta(in(ii))
         sum2=sum2+delta(in(ii))**2
         sum4=sum4+delta(in(ii))**4
       enddo
c
c  derivatives of axb wrt hj:
c
       daxb(j,1,1)=0.0d0
       daxb(j,1,2)=b(3)-a(3)
       daxb(j,1,3)=-b(2)+a(2)
       daxb(j,2,1)=-b(3)+a(3)
       daxb(j,2,2)=0.0d0
       daxb(j,2,3)=b(1)-a(1)
       daxb(j,3,1)=b(2)-a(2)
       daxb(j,3,2)=-b(1)+a(1)
       daxb(j,3,3)=0.0d0
c
c  derivatives of axb wrt hk:
c
       daxb(k,1,1)=0.0d0
       daxb(k,1,2)=-b(3)
       daxb(k,1,3)=b(2)
       daxb(k,2,1)=b(3)
       daxb(k,2,2)=0.0d0
       daxb(k,2,3)=-b(1)
       daxb(k,3,1)=-b(2)
       daxb(k,3,2)=b(1)
       daxb(k,3,3)=0.0d0
c
c  derivatives of axb wrt hl:
c
       daxb(l,1,1)=0.0d0
       daxb(l,1,2)=a(3)
       daxb(l,1,3)=-a(2)
       daxb(l,2,1)=-a(3)
       daxb(l,2,2)=0.0d0
       daxb(l,2,3)=a(1)
       daxb(l,3,1)=a(2)
       daxb(l,3,2)=-a(1)
       daxb(l,3,3)=0.0d0
c
c   loop over cdot(in(ii),ind,jind) where we consider deriv of c(in(ii))
c   wrt h(in(ii),jind) with components jind
c
       do ii=1,3
c
c  deriv of cdot(in(ii),x) wrt x, y, z
c
         cdot(in(ii),1,1)=1.0d0/rch(in(ii))
     *                   +tch(in(ii),1)*c(in(ii),1)/rch(in(ii))**2
         cdot(in(ii),1,2)=tch(in(ii),2)*c(in(ii),1)/rch(in(ii))**2
         cdot(in(ii),1,3)=tch(in(ii),3)*c(in(ii),1)/rch(in(ii))**2
c
c  deriv of cdot(in(ii),y) wrt x, y, z
c
         cdot(in(ii),2,1)=tch(in(ii),1)*c(in(ii),2)/rch(in(ii))**2
         cdot(in(ii),2,2)=1.0d0/rch(in(ii))
     *                   +tch(in(ii),2)*c(in(ii),2)/rch(in(ii))**2
         cdot(in(ii),2,3)=tch(in(ii),3)*c(in(ii),2)/rch(in(ii))**2
c
c  deriv of cdot(in(ii),z) wrt x, y, z
c
         cdot(in(ii),3,1)=tch(in(ii),1)*c(in(ii),3)/rch(in(ii))**2
         cdot(in(ii),3,2)=tch(in(ii),2)*c(in(ii),3)/rch(in(ii))**2
         cdot(in(ii),3,3)=1.0d0/rch(in(ii))
     *                   +tch(in(ii),3)*c(in(ii),3)/rch(in(ii))**2
       enddo
c
       do ii=1,3
         do ind=1,3
            deldot=-dtheta0(i,in(ii),i)
c
c  derivative wrt h(i,ind)
c  for  rch(i) only terms are from the derivatives of theta0
c
            deldot=-deldot*tch(i,ind)/rch(i)
            pdot(nh(i,ind))=pdot(nh(i,ind))
     *                   +2.0d0*fdelta(i)*delta(in(ii))*deldot
     *                   +4.0d0*hdelta(i)*delta(in(ii))**3*deldot
c
c  derivative wrt c(ind)
c
            deldot=-deldot
            pdot(nc(ind))=pdot(nc(ind))
     *                   +2.0d0*fdelta(i)*delta(in(ii))*deldot
     *                   +4.0d0*hdelta(i)*delta(in(ii))**3*deldot
           do jj=1,3
c
c  partial derivatives wrt h(in(jj),ind), loop over delta(i,in(ii))
c
c   atemp1 is axb dot daxb wrt h(in(jj))
c
            atemp1=axb(1)*daxb(in(jj),ind,1)
     *            +axb(2)*daxb(in(jj),ind,2)
     *            +axb(3)*daxb(in(jj),ind,3)
            atemp1=atemp1/(norma**3)
c
c  atemp2 is deriv of normalised axb
c
            atemp2(1)=daxb(in(jj),ind,1)/norma-atemp1*axb(1)
            atemp2(2)=daxb(in(jj),ind,2)/norma-atemp1*axb(2)
            atemp2(3)=daxb(in(jj),ind,3)/norma-atemp1*axb(3)
c
c  atemp3 is daxb dot c(in(ii))

            atemp3=atemp2(1)*c(in(ii),1)+atemp2(2)*c(in(ii),2)
     *                             +atemp2(3)*c(in(ii),3)
c
c  atemp4 is axb dot cdot
c
            atemp4=0.0d0
            if(ii.eq.jj)then
c
c  ie deriv of c(in(ii)) wrt h(in(jj)) is non zero only for ii = jj
c
              atemp4=axb(1)*cdot(in(ii),1,ind)
     *                     +axb(2)*cdot(in(ii),2,ind)
     *                     +axb(3)*cdot(in(ii),3,ind)
              atemp4=atemp4/norma
            endif
c
c  atemp5 is deriv of theta0(i,in(ii)) wrt to nh(in(jj),ind)
c
            atemp5=-dtheta0(i,in(ii),in(jj))
c
c  deriv wrt h(in(jj)),ind):
c
            atemp5=-atemp5*tch(in(jj),ind)/rch(in(jj))
            deldot=atemp3+atemp4
            deldot=-1.0d0/sqrt(1.0d0-argd(in(ii))**2)*deldot
            deldot=deldot+atemp5
            pdot(nh(in(jj),ind))=pdot(nh(in(jj),ind))
     *                   +2.0d0*fdelta(i)*delta(in(ii))*deldot
     *                   +4.0d0*hdelta(i)*delta(in(ii))**3*deldot
c
c  for carbon the only contributions are from axb dot cdot term and
c  from theta0 and derivative cdot wrt carbon=-cdot wrt hydrogen
c
            deldot=1.0d0/sqrt(1.0d0-argd(in(ii))**2)*atemp4
            deldot=deldot-atemp5
            pdot(nc(ind))=pdot(nc(ind))
     *                   +2.0d0*fdelta(i)*delta(in(ii))*deldot
     *                   +4.0d0*hdelta(i)*delta(in(ii))**3*deldot
          enddo
        enddo
       enddo
       return
       end

c******************************************************
c
       subroutine opforce
c
c  calculates the out-of-plane bending force constants
c  and their derivatives
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
c
       dimension switch(4),dswitch(4,4)
c
c  calculate switching functions:
c
       switch(1)=(1.0d0-s3(1))*s3(2)*s3(3)*s3(4)
       switch(2)=(1.0d0-s3(2))*s3(3)*s3(4)*s3(1)
       switch(3)=(1.0d0-s3(3))*s3(4)*s3(1)*s3(2)
       switch(4)=(1.0d0-s3(4))*s3(1)*s3(2)*s3(3)
c
c  calculate derivatives:
c  derivative of switch(1) wrt the 4 rch bond lengths
c
       dswitch(1,1)=-ds3(1)*s3(2)*s3(3)*s3(4)
       dswitch(1,2)=(1.0d0-s3(1))*ds3(2)*s3(3)*s3(4)
       dswitch(1,3)=(1.0d0-s3(1))*s3(2)*ds3(3)*s3(4)
       dswitch(1,4)=(1.0d0-s3(1))*s3(2)*s3(3)*ds3(4)
c
c  derivative of switch(2) wrt the 4 rch bond lengths
c
       dswitch(2,1)=(1.0d0-s3(2))*s3(3)*s3(4)*ds3(1)
       dswitch(2,2)=-ds3(2)*s3(3)*s3(4)*s3(1)
       dswitch(2,3)=(1.0d0-s3(2))*ds3(3)*s3(4)*s3(1)
       dswitch(2,4)=(1.0d0-s3(2))*s3(3)*ds3(4)*s3(1)
c
c  derivative of switch(3) wrt the 4 rch bond lengths
c
       dswitch(3,1)=(1.0d0-s3(3))*s3(4)*ds3(1)*s3(2)
       dswitch(3,2)=(1.0d0-s3(3))*s3(4)*s3(1)*ds3(2)
       dswitch(3,3)=-ds3(3)*s3(4)*s3(1)*s3(2)
       dswitch(3,4)=(1.0d0-s3(3))*ds3(4)*s3(1)*s3(2)
c
c  derivative of switch(3) wrt the 4 rch bond lengths
c
       dswitch(4,1)=(1.0d0-s3(4))*ds3(1)*s3(2)*s3(3)
       dswitch(4,2)=(1.0d0-s3(4))*s3(1)*ds3(2)*s3(3)
       dswitch(4,3)=(1.0d0-s3(4))*s3(1)*s3(2)*ds3(3)
       dswitch(4,4)=-ds3(4)*s3(1)*s3(2)*s3(3)
c
c  calculate the force constants and their derivatives
c
       do i=1,4
         fdelta(i)=switch(i)*fch3
         hdelta(i)=switch(i)*hch3
         do j=1,4
           dfdelta(i,j)=dswitch(i,j)*fch3
           dhdelta(i,j)=dswitch(i,j)*hch3
         enddo
       enddo
       return
       end
c
c******************************************************
c
c
       subroutine ipforce
c
c  calculates the symmetrised in plane bend force constants and
c  all partial derivatives involving them
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
       dimension dfk0(4,4,4),df1dc(4),df1dh(4)
c
c  set force constant at asymptotes
c
       f0=fkinf+ak
       f2=fkinf
c
       fk0(1,2)=f0+f0*(s1(1)*s1(2)-1.0d0)+(f0-f2)*(s2(3)*s2(4)-1.0d0)
       fk0(1,3)=f0+f0*(s1(1)*s1(3)-1.0d0)+(f0-f2)*(s2(2)*s2(4)-1.0d0)
       fk0(1,4)=f0+f0*(s1(1)*s1(4)-1.0d0)+(f0-f2)*(s2(2)*s2(3)-1.0d0)
       fk0(2,3)=f0+f0*(s1(2)*s1(3)-1.0d0)+(f0-f2)*(s2(1)*s2(4)-1.0d0)
       fk0(2,4)=f0+f0*(s1(2)*s1(4)-1.0d0)+(f0-f2)*(s2(1)*s2(3)-1.0d0)
       fk0(3,4)=f0+f0*(s1(3)*s1(4)-1.0d0)+(f0-f2)*(s2(1)*s2(2)-1.0d0)
c
c  derivative of fk0
c
       dfk0(1,2,1)=f0*ds1(1)*s1(2)
       dfk0(1,2,2)=f0*s1(1)*ds1(2)
       dfk0(1,2,3)=(f0-f2)*ds2(3)*s2(4)
       dfk0(1,2,4)=(f0-f2)*s2(3)*ds2(4)
c
       dfk0(1,3,1)=f0*ds1(1)*s1(3)
       dfk0(1,3,2)=(f0-f2)*ds2(2)*s2(4)
       dfk0(1,3,3)=f0*s1(1)*ds1(3)
       dfk0(1,3,4)=(f0-f2)*s2(2)*ds2(4)
c
       dfk0(1,4,1)=f0*ds1(1)*s1(4)
       dfk0(1,4,2)=(f0-f2)*ds2(2)*s2(3)
       dfk0(1,4,3)=(f0-f2)*s2(2)*ds2(3)
       dfk0(1,4,4)=f0*s1(1)*ds1(4)
c
       dfk0(2,3,1)=(f0-f2)*ds2(1)*s2(4)
       dfk0(2,3,2)=f0*ds1(2)*s1(3)
       dfk0(2,3,3)=f0*s1(2)*ds1(3)
       dfk0(2,3,4)=(f0-f2)*s2(1)*ds2(4)
c
       dfk0(2,4,1)=(f0-f2)*ds2(1)*s2(3)
       dfk0(2,4,2)=f0*ds1(2)*s1(4)
       dfk0(2,4,3)=(f0-f2)*s2(1)*ds2(3)
       dfk0(2,4,4)=f0*s1(2)*ds1(4)
c
       dfk0(3,4,1)=(f0-f2)*ds2(1)*s2(2)
       dfk0(3,4,2)=(f0-f2)*s2(1)*ds2(2)
       dfk0(3,4,3)=f0*ds1(3)*s1(4)
       dfk0(3,4,4)=f0*s1(3)*ds1(4)
c
c       argfk0=bk*((rch(1)-r0ch)**2+(rch(2)-r0ch)**2
c     *              +(rch(3)-r0ch)**2+(rch(4)-r0ch)**2)
c       fk0=fkinf+ak*exp(-argfk0)
       do i=1,4
c
c  calc derivatives of fk0 wrt each of the rch(i) bonds
c
c         dfk0(i)=-2.0d0*ak*bk*(rch(i)-r0ch)*exp(-argfk0)
c
c  calculate the terms f1(i)
c
         arga1=aa1*rbh(i)*rbh(i)
         arga2=aa4*(rbh(i)-r0hh)*(rbh(i)-r0hh)
         a1=1.0d0-exp(-arga1)
         a2=aa2+aa3*exp(-arga2)
         f1(i)=a1*exp(-a2*(rch(i)-r0ch)**2)
c
c  and calculate the derivatives wrt rch(i) and rbh(i)
c
         duma1=2.0d0*aa1*rbh(i)*exp(-arga1)
         duma2=-2.0d0*aa3*aa4*(rbh(i)-r0hh)*exp(-arga2)
         df1dc(i)=-2.0d0*(rch(i)-r0ch)*a1*a2*exp(-a2*(rch(i)-r0ch)**2)
         df1dh(i)=duma1*exp(-a2*(rch(i)-r0ch)**2)
     *             -duma2*(rch(i)-r0ch)**2*a1*exp(-a2*(rch(i)-r0ch)**2)
       enddo
c
c  derivative of total force constant f(i,j) wrt bond length rch(k)
c  is given by dfdc(i,j,k)
c
      dfdc(1,2,1)=dfk0(1,2,1)*f1(1)*f1(2)+fk0(1,2)*df1dc(1)*f1(2)
      dfdc(1,2,2)=dfk0(1,2,2)*f1(1)*f1(2)+fk0(1,2)*f1(1)*df1dc(2)
      dfdc(1,2,3)=dfk0(1,2,3)*f1(1)*f1(2)
      dfdc(1,2,4)=dfk0(1,2,4)*f1(1)*f1(2)
c
      dfdc(1,3,1)=dfk0(1,3,1)*f1(1)*f1(3)+fk0(1,3)*df1dc(1)*f1(3)
      dfdc(1,3,2)=dfk0(1,3,2)*f1(1)*f1(3)
      dfdc(1,3,3)=dfk0(1,3,3)*f1(1)*f1(3)+fk0(1,3)*f1(1)*df1dc(3)
      dfdc(1,3,4)=dfk0(1,3,4)*f1(1)*f1(3)
c
      dfdc(1,4,1)=dfk0(1,4,1)*f1(1)*f1(4)+fk0(1,4)*df1dc(1)*f1(4)
      dfdc(1,4,2)=dfk0(1,4,2)*f1(1)*f1(4)
      dfdc(1,4,3)=dfk0(1,4,3)*f1(1)*f1(4)
      dfdc(1,4,4)=dfk0(1,4,4)*f1(1)*f1(4)+fk0(1,4)*f1(1)*df1dc(4)
c
      dfdc(2,3,1)=dfk0(2,3,1)*f1(2)*f1(3)
      dfdc(2,3,2)=dfk0(2,3,2)*f1(2)*f1(3)+fk0(2,3)*df1dc(2)*f1(3)
      dfdc(2,3,3)=dfk0(2,3,3)*f1(2)*f1(3)+fk0(2,3)*f1(2)*df1dc(3)
      dfdc(2,3,4)=dfk0(2,3,4)*f1(2)*f1(3)
c
      dfdc(2,4,1)=dfk0(2,4,1)*f1(2)*f1(4)
      dfdc(2,4,2)=dfk0(2,4,2)*f1(2)*f1(4)+fk0(2,4)*df1dc(2)*f1(4)
      dfdc(2,4,3)=dfk0(2,4,3)*f1(2)*f1(4)
      dfdc(2,4,4)=dfk0(2,4,4)*f1(2)*f1(4)+fk0(2,4)*f1(2)*df1dc(4)
c
      dfdc(3,4,1)=dfk0(3,4,1)*f1(3)*f1(4)
      dfdc(3,4,2)=dfk0(3,4,2)*f1(3)*f1(4)
      dfdc(3,4,3)=dfk0(3,4,3)*f1(3)*f1(4)+fk0(3,4)*df1dc(3)*f1(4)
      dfdc(3,4,4)=dfk0(3,4,4)*f1(3)*f1(4)+fk0(3,4)*f1(3)*df1dc(4)
c
c  derivative of total force constant f(i,j) wrt bond length rbh(k)
c  is given by dfdh(i,j,k)
c
c  nb only non-zero derivatives are those from rbh(i) and rbh(j)
c
       dfdh(1,2,1)=fk0(1,2)*df1dh(1)*f1(2)
       dfdh(1,2,2)=fk0(1,2)*f1(1)*df1dh(2)
       dfdh(1,2,3)=0.0d0
       dfdh(1,2,4)=0.0d0
c
       dfdh(1,3,1)=fk0(1,3)*df1dh(1)*f1(3)
       dfdh(1,3,2)=0.0d0
       dfdh(1,3,3)=fk0(1,3)*f1(1)*df1dh(3)
       dfdh(1,3,4)=0.0d0
c
       dfdh(1,4,1)=fk0(1,4)*df1dh(1)*f1(4)
       dfdh(1,4,2)=0.0d0
       dfdh(1,4,3)=0.0d0
       dfdh(1,4,4)=fk0(1,4)*f1(1)*df1dh(4)
c
       dfdh(2,3,1)=0.0d0
       dfdh(2,3,2)=fk0(2,3)*df1dh(2)*f1(3)
       dfdh(2,3,3)=fk0(2,3)*f1(2)*df1dh(3)
       dfdh(2,3,4)=0.0d0
c
       dfdh(2,4,1)=0.0d0
       dfdh(2,4,2)=fk0(2,4)*df1dh(2)*f1(4)
       dfdh(2,4,3)=0.0d0
       dfdh(2,4,4)=fk0(2,4)*f1(2)*df1dh(4)
c
       dfdh(3,4,1)=0.0d0
       dfdh(3,4,2)=0.0d0
       dfdh(3,4,3)=fk0(3,4)*df1dh(3)*f1(4)
       dfdh(3,4,4)=fk0(3,4)*f1(3)*df1dh(4)
c
       return
       end
c
c******************************************************
c
c
       subroutine switchf
c
c  calculates switching functions: s3,sphi,stheta
c  and their derivatives ds3,dsphi,dstheta
c
       implicit double precision (a-h,o-z)
       include 'ch4cn.inc'
c
c  nb remember that integration units are:
c  energy in   1.0d+05 j/mol
c  time in     1.0d-14 s
c
       a1s=1.5313681d-7
       b1s=-4.6696246d0
       a2s=1.0147402d-7
       b2s=-12.363798d0
c
c  use double precision criterion:
c
c  tanh(19.0d0)=1.0d0
c
       argmax=19.0d0
       do i=1,4
         args1=a1s*(rch(i)-r0ch)*(rch(i)-b1s)**8
         if(args1.lt.argmax)then
           s1(i)=1.0d0-tanh(args1)
           ds1(i)=a1s*((rch(i)-b1s)**8
     *                 +8.0d0*(rch(i)-r0ch)*(rch(i)-b1s)**7)
           ds1(i)=-ds1(i)/cosh(args1)**2
         else
           s1(i)=0.0d0
           ds1(i)=0.0d0
         endif
c
         args2=a2s*(rch(i)-r0ch)*(rch(i)-b2s)**6
         if(args2.lt.argmax)then
           s2(i)=1.0d0-tanh(args2)
           ds2(i)=a2s*((rch(i)-b2s)**6
     *                 +6.0d0*(rch(i)-r0ch)*(rch(i)-b2s)**5)
           ds2(i)=-ds2(i)/cosh(args2)**2
         else
           s2(i)=0.0d0
           ds2(i)=0.0d0
         endif
c
c  calculate s3 and ds3
c
         args3=a3s*(rch(i)-r0ch)*(rch(i)-b3s)**2
         if (args3.lt.argmax)then
           s3(i)=1.0d0-tanh(args3)
           ds3(i)=a3s*(3.0d0*rch(i)**2-2.0d0*rch(i)*(r0ch+2.0d0*b3s)
     *          +b3s*(b3s+2.0d0*r0ch))
           ds3(i)=-ds3(i)/cosh(args3)**2
         else
           s3(i)=0.0d0
           ds3(i)=0.0d0
         endif
c
c  calculate sphi and dsphi
c
c  condition here is on the bondlength rch(i)
c  st argsphi is lt approx 19.0d0
c
         if(rch(i).lt.3.8d0)then
           argsphi=aphi*(rch(i)-r0ch)*exp(bphi*(rch(i)-cphi)**3)
           sphi(i)=1.0d0-tanh(argsphi)
           dsphi(i)=aphi*(1.0d0+3.0d0*bphi*(rch(i)-r0ch)
     *                      *(rch(i)-cphi)**2)
           dsphi(i)=dsphi(i)*exp(bphi*(rch(i)-cphi)**3)
           dsphi(i)=-dsphi(i)/cosh(argsphi)**2
         else
           sphi(i)=0.0d0
           dsphi(i)=0.0d0
         endif
c
c  calculate stheta and dstheta
c
         if(rch(i).lt.3.8d0)then
           argstheta=atheta*(rch(i)-r0ch)*exp(btheta*(rch(i)-ctheta)**3)
           stheta(i)=1.0d0-tanh(argstheta)
           dstheta(i)=atheta*(1.0d0+3.0d0*btheta*(rch(i)-r0ch)
     *           *(rch(i)-ctheta)**2)
           dstheta(i)=dstheta(i)*exp(btheta*(rch(i)-ctheta)**3)
           dstheta(i)=-dstheta(i)/cosh(argstheta)**2
         else
           stheta(i)=0.0d0
           dstheta(i)=0.0d0
         endif
       enddo
       return
       end
C
      SUBROUTINE SETUP(N3TM)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C   N3TMMN = 3 * NATOMS
C   NATOMS = the number of atoms represented by this potential function
C
C   The variable N3TMMN is the minimum value of N3TM allowed to be 
C   passed by the calling routine for the number of cartesian 
C   coordinates needed to represent the full system represented by this 
C   potential energy surface routine.
C   N3TM must be greater than or equal to N3TMMN.
C
      PARAMETER (N3TMMN = 18)
C
      COMMON /PDATCM/ D1(3),D3(3),ALPH(3),RE(3),BETA(3),CC(3),AA(3),    
     *   APARM(5),REFV,TAU,CP,B1,C1
      COMMON /PDT2CM/ PI,APHI,BPHI,CPHI,PCH4,FCH3,HCH3,RNOT,A3,B3,DELT, 
     *   DIJ,A,B,C0,D,REX,ATETHA,BTETHA,CTETHA
      COMMON /PDT3CM/ FK0,FA,CA,CB
      CHARACTER*5 SURFNM
C
C  CHECK THE NUMBER OF CARTESIAN COORDINATES SET BY THE CALLING PROGRAM
C
      IF (N3TM .LT. N3TMMN) THEN
          WRITE (6, 6000) N3TM, N3TMMN
          STOP 'SETUP 1'
      ENDIF
C
      RETURN
C
6000  FORMAT(/,2X,T5,'Warning: N3TM is set equal to ',I3,
     *                  ' but this potential routine',
     *          /,2X,T14,'requires N3TM be greater than or ',
     *                   'equal to ',I3,/)
C
      END
