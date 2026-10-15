module parm
implicit none
integer, parameter :: n_en_member=200,no_sta_var=2,no_obs_var=1
integer :: da_daycount=1,cnt_strt=1,cnt_end=200,wrdtf=0
real :: ens_sol_st(200,2,1),ens_sol_st_1(200,2,1),sol_z(2,1),sol_wpmm(2,1)
real :: xbb(2,200),xbb_1(2,200),xaa(2,200),y(1,200),rand_da_en(200,1)
real :: smobs(1,1),sm_dqx(1,1),sm_rfi(1,1),sm_dqx_co=1,sm_rfi_co=1
real :: sm_std_obs_all(1,1),ens_samp(200,6),err_xb(2,200),err_y(1,200)
real :: st_er_cov(2,2),st_par_cov(2,1),st_obs_cov(2,1),obs_er_cov(1,1)
real :: kalman_gain(2,1),sol_fc(2,1),sol_ul(2,1),KG_all(1,1),inflow_flag(1,1)=0
real :: audit_raw_increment,audit_projected_increment
integer :: audit_clip_low,audit_clip_high
real :: sol_st(2,1),ens_sol_sw(200,1),sol_sw(1),da_flag(1,1)
integer :: sol_nly(1)=2
end module
