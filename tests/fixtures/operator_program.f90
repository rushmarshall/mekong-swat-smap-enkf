program test_operator
use parm
implicit none
integer :: k,case_id
real :: prior(2),expected(2),denom,cov(2),a(200),eps(200),target
real :: expected_xaa(2,200),raw_increment,projected_increment
sol_z(:,1)=(/50.0,495.0/)
sol_wpmm(:,1)=(/5.0,44.5/)
sol_fc(:,1)=(/15.0,133.5/)
sol_ul(:,1)=(/20.0,160.0/)
do k=1,200
  a(k)=(real(k)-100.5)/50.0
  eps(k)=sin(real(k)*0.91)
enddo
eps=eps-sum(eps)/200.0
eps=eps/sqrt(sum(eps**2)/199.0)
ens_samp=0.0
ens_samp(:,2)=eps
do case_id=1,5
  ens_sol_st(:,1,1)=10.0+a
  ens_sol_st(:,2,1)=40.0+2.0*a
  ens_sol_st_1=ens_sol_st
  if(case_id==1) target=0.30
  if(case_id==2) target=0.34
  if(case_id==3) target=0.06
  if(case_id==4) target=-1.0
  if(case_id==5) target=2.0
  smobs(1,1)=target
  call init_da(1)
  if(abs(sm_std_obs_all(1,1)-2.0)>1.e-6) stop 11
  ! Accumulate in double precision when checking a single-precision routine.
  if(abs(sum(dble(y))/200.d0-dble(target*50.0-5.0))>1.e-5) stop 12
  prior=sum(xbb,2)/200.0
  cov(1)=sum((xbb(1,:)-prior(1))**2)/199.0
  cov(2)=sum((xbb(2,:)-prior(2))*(xbb(1,:)-prior(1)))/199.0
  denom=cov(1)+sum((y(1,:)-sum(y)/200.0)**2)/199.0
  do k=1,200
    expected_xaa(:,k)=xbb(:,k)+cov/denom*(y(1,k)-xbb(1,k))
  enddo
  raw_increment=sum(expected_xaa-xbb)/200.0
  do k=1,200
    expected_xaa(:,k)=max(0.0,min(sol_ul(:,1),expected_xaa(:,k)))
  enddo
  projected_increment=sum(expected_xaa-xbb)/200.0
  expected=sum(expected_xaa,2)/200.0
  call EnKF(1)
  if(maxval(abs(sum(xaa,2)/200.0-expected))>2.e-4) stop 13
  call post_da(1)
  if(minval(sol_st)<0.0) stop 14
  if(any(ens_sol_st(:,1,1)>20.0).or.any(ens_sol_st(:,2,1)>160.0)) stop 17
  if(any(ens_sol_st<0.0)) stop 18
  if(abs(audit_raw_increment-raw_increment)>2.e-4) stop 19
  if(abs(audit_projected_increment-projected_increment)>2.e-4) stop 20
  if(case_id==4.and.audit_clip_low==0) stop 21
  if(case_id==5.and.audit_clip_high==0) stop 22
  if(abs(sol_sw(1)-sum(sol_st(:,1)))>1.e-4) stop 15
  if(case_id==1.and.abs(sol_st(1,1)-10.0)>2.e-5) stop 16
  write(*,*)case_id,target,prior,sol_st(:,1),kalman_gain(:,1)
enddo
print *, 'PASS: unit transform, error variance, no-change target, increment, bounds, water total'
end program
