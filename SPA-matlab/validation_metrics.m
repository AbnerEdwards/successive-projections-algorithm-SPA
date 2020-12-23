function [PRESS,RMSEP,SDV,BIAS,r] = validation_metrics(Xcal,ycal,Xval,yval,var_sel)

% [PRESSV,RMSEPV,SDV,BIASV,rV] = validation_metrics(Xcal,ycal,Xval,yval,var_sel) --> Validation with a separate set
% [PRESSCV,RMSEPCV,SDCV,BIASCV,rCV] = validation_metrics(Xcal,ycal,[],[],var_sel) --> Cross-validation

if size(Xval,1) > 0 % Validation with a separate set
    y = yval;
else % Cross-validation
    y = ycal;
end

[yhat,e] = validation(Xcal,ycal,Xval,yval,var_sel);

PRESS = e'*e;
N = length(e);
RMSEP = sqrt(PRESS/N);
BIAS = mean(e);
ec = e - BIAS; % Mean-centered error values
SDV = sqrt(ec'*ec/(N - 1));
yhat_as = (yhat - mean(yhat))/std(yhat); % Autoscaling
y_as = (y - mean(y))/std(y); % Autoscaling
r = (yhat_as'*y_as)/(N-1);

% Statistical Prediction Errors
spe = statistical_prediction_error(Xcal,ycal,Xval,yval,var_sel);

% Plot of Predicted vs Reference values
figure, hold on, grid
errorbar(y,yhat,spe,'o')
% plot(y,yhat,'o')
xlabel('Reference y value'),ylabel('Predicted y value')
h = gca; XLim = get(h,'XLim');
h = line(XLim,Xlim);
title(['PRESS = ' num2str(PRESS) ', RMSEP = ' num2str(RMSEP) ', SDV = ' num2str(SDV) ', BIAS = ' num2str(BIAS) ', r = ' num2str(r)])



