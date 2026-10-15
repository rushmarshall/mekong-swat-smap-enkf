subroutine init_da(jj)
    use parm
    real :: interm_calc (n_en_member,no_sta_var), sm_temp,sm_temp2, sm_std_obs
    integer :: ii
    integer, intent(in) :: jj
    interm_calc(1:n_en_member,1)=ens_sol_st(1:n_en_member,1,jj)
    interm_calc(1:n_en_member,2)=ens_sol_st(1:n_en_member,2,jj)
    
    xbb = transpose(interm_calc)
    
    interm_calc(1:n_en_member,1)=ens_sol_st_1(1:n_en_member,1,jj)
    interm_calc(1:n_en_member,2)=ens_sol_st_1(1:n_en_member,2,jj)
    
    xbb_1 = transpose(interm_calc)
    
    
    ! Volumetric observation -> available layer water, millimetres.
    ! SWAT sol_st excludes water held below wilting point.
    sm_temp = smobs(da_daycount,jj)*sol_z(1,jj)-sol_wpmm(1,jj)
    ! Constant 0.04 m3/m3 transformed through the same operator.
    sm_std_obs = 0.04*sol_z(1,jj)
    sm_std_obs_all(da_daycount,jj)= sm_std_obs
    
    rand_da_en (1:n_en_member,1) = ens_samp(cnt_strt:cnt_end,2)
    do ii = 1,n_en_member
    y(1,ii)=((sm_temp+((rand_da_en(ii,1)*sm_std_obs))))
    end do
    
    
    
    end subroutine