subroutine EnKF(jj)
    !this subroutine calculates analysis vector using state vector and observation vector in EnKF
    
    use parm
    real :: innovation (no_obs_var,n_en_member),innovation_mean(no_obs_var), y_mean,rd_mean,rd_var,kk,rd_std, y_var, y_std, dummy1(1,n_en_member), ii, dummy22
    real :: xbb_mean1, xbb_var1, xbb_std1, xbb_mean2, xbb_var2, xbb_std2
    real :: xaa_mean (no_sta_var,1), xaa_var1, xaa_std1, xaa_mean2, xaa_var2, xaa_std2, xaa_mean1
    integer, intent(in) :: jj   !hru no
    integer, save :: audit_lastday=-1
    
    ! Calculate State Error and Observation Error Vector and respective covariances
    call st_error
    call obs_error
    call st_cov
    call obs_cov
    call st_obs
    
    ! Calculate Kalman gain Function
    
    call Kalman_Gain_calc
    do ii = 1,n_en_member
    dummy1(1,ii) = xbb(1,ii)
    innovation(1,ii) = y(1,ii)-dummy1(1,ii)
    end do
    
    innovation_mean = sum(innovation(1,1:n_en_member))/n_en_member
    xaa = xbb + matmul(kalman_gain,innovation)
    xaa_mean(:,1) = sum(xaa,2)/size(xaa,2)
    ! sol_ul is saturated water minus wilting-point water, in mm.
    ! Preserve both proposals and physically projected mass adjustments.
    audit_raw_increment = sum(xaa-xbb)/real(n_en_member)
    audit_clip_low = 0
    audit_clip_high = 0
    do ii = 1,no_sta_var
        audit_clip_low = audit_clip_low + count(xaa(ii,:) < 0.0)
        audit_clip_high = audit_clip_high + count(xaa(ii,:) > sol_ul(ii,jj))
        xaa(ii,:) = max(0.0,min(sol_ul(ii,jj),xaa(ii,:)))
    end do
    audit_projected_increment = sum(xaa-xbb)/real(n_en_member)
    xaa_mean(:,1) = sum(xaa,2)/size(xaa,2)

    KG_all(da_daycount,jj) = kalman_gain(1,1)
    if (audit_lastday /= da_daycount) then
      open(992,file='operator_sample.csv')
      write(992,9921) da_daycount,jj,sum(xbb(1,:))/n_en_member, &
        st_er_cov(1,1),sum(y)/n_en_member,sm_std_obs_all(da_daycount,jj), &
        kalman_gain(1,1),sum(xaa(1,:))/n_en_member, &
        audit_raw_increment,audit_projected_increment,audit_clip_low,audit_clip_high
9921  format(i8,',',i8,8(',',es16.8),2(',',i8))
      audit_lastday=da_daycount
    end if    
    
    
       
    if (jj == 1 .and. wrdtf == 1) then
!%%  xbb
    ! 1st layer
    xbb_mean1 = sum(xbb(1,1:n_en_member))/n_en_member
    xbb_var1 = 0
    do kk=1, n_en_member
         xbb_var1 = xbb_var1 + (((xbb(1,kk)-xbb_mean1)**2.0)/n_en_member)
    end do
    xbb_std1 = xbb_var1**0.5
    ! 2nd layer
    xbb_mean2 = sum(xbb(2,1:n_en_member))/n_en_member
    xbb_var2 = 0
    do kk=1, n_en_member
         xbb_var2 = xbb_var2 + (((xbb(2,kk)-xbb_mean2)**2.0)/n_en_member)
    end do
    xbb_std2 = xbb_var2**0.5
    
!%%  xaa 
    ! 1st layer
    xaa_mean1 = sum(xaa(1,1:n_en_member))/n_en_member
    xaa_var1 = 0
    do kk=1, n_en_member
         xaa_var1 = xaa_var1 + (((xaa(1,kk)-xaa_mean1)**2.0)/n_en_member)
    end do
    xaa_std1 = xaa_var1**0.5
    ! 2nd layer
    xaa_mean2 = sum(xaa(2,1:n_en_member))/n_en_member
    xaa_var2 = 0
    do kk=1, n_en_member
         xaa_var2 = xaa_var2 + (((xaa(2,kk)-xaa_mean2)**2.0)/n_en_member)
    end do
    xaa_std2 = xaa_var2**0.5
    
!%%  y  
    y_mean =sum(y)/n_en_member
    y_var = 0
    do kk=1, n_en_member
         y_var = y_var + (((y(1,kk)-y_mean)**2.0)/n_en_member)
    end do
    y_std = y_var**0.5
    
    
    rd_mean=sum(rand_da_en)/n_en_member
    rd_var = 0
    do kk=1, n_en_member
         rd_var = rd_var + (((rand_da_en(kk,1)-rd_mean)**2.0)/n_en_member)
    end do
    rd_std = sqrt(rd_var)

    
    open (521,file='KalmanGain.dat')
    write (521,1234) da_daycount, inflow_flag(da_daycount,jj),innovation_mean, kalman_gain(1,1), kalman_gain(2,1),xbb_mean1, xbb_mean2, xaa_mean1, xaa_mean2, xbb_std1, xbb_std2, xaa_std1, xaa_std2, y_mean, y_std, st_par_cov(1,1),st_par_cov(2,1)
    

    end if

1234 format(3x,i4,3x, 16(f11.5))
     
    
    
    
    end 