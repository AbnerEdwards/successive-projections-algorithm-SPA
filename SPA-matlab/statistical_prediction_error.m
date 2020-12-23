function spe = statistical_prediction_error(Xcal,ycal,Xval,yval,var_sel)

N = size(Xcal,1); % Number of objects in the calibration set
NV = size(Xval,1); % Number of objects in the validation set
m = length(var_sel);

Xcal_ones = [ones(length(ycal),1) Xcal(:,var_sel)];
b = Xcal_ones\ycal; % MLR with offset term (b0)
ecal = ycal - Xcal_ones*b; % Regression residuals 

s2 = ecal'*ecal/(N - m - 1);

if NV > 0 % Validation with a separate set
    Xcal_ones = [ones(N,1) Xcal(:,var_sel)];
    Xval_ones = [ones(NV,1) Xval(:,var_sel)];
    spe = sqrt(s2*diag(Xval_ones * inv(Xcal_ones'*Xcal_ones) * Xval_ones'));
else % Cross-validation    
	for i = 1:N
       % Removing the ith object from the calibration set
       cal = [[1:i-1] [i+1:N]];
       X = Xcal(cal,var_sel);
       y = ycal(cal);
       xtest = [1 Xcal(i,var_sel)];
       ytest = ycal(i);
       X_ones = [ones(N-1,1) X];
       spe(i) = sqrt(s2*xtest * inv(X_ones'*X_ones) * xtest');
    end
end