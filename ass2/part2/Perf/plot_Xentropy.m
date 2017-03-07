load('PR_Xentropy_gru3l32u.mat')
PRX = mat;
load('GT_Xentropy_gru3l32u.mat')
GTX = mat;

meangt = mean(GTX,1);
meanpr = mean(PRX,1);

[N D] = size(GTX);
samples = 1:N;
one = ones(N,1);


gw = 0.9 ; % almost touching, 1 is touching
f1 = bar(samples-gw/6,GTX(:,1),gw/2,'b','EdgeColor','none') ;
xlim([0 101])
%ylim([0 1.1])
title('Xentropy for next n=300 pixels')
xlabel('sample') % x-axis label
ylabel('Xentropy') % y-axis label
hold on ;
f2 = bar(samples+gw/6,PRX(:,1),gw/2,'r','EdgeColor','none') ;
hold on
plot(samples,one*mean(GTX(:,1)),'b',samples,one*mean(PRX(:,1)),'r','LineWidth',1);
legend('GT Xentropy','PredictedXentropy','average GT Xentropy','average Predicted Xentropy','Location','best')
hold off

gw = 0.9 ; % almost touching, 1 is touching
f1 = bar(samples-gw/6,GTX(:,2),gw/2,'b','EdgeColor','none') ;
xlim([0 101])
ylim([0 0.72])
title('Xentropy for next n=28 pixels')
xlabel('sample') % x-axis label
ylabel('Xentropy') % y-axis label
hold on ;
f2 = bar(samples+gw/6,PRX(:,2),gw/2,'r','EdgeColor','none') ;
hold on
plot(samples,one*mean(GTX(:,2)),'b',samples,one*mean(PRX(:,2)),'r','LineWidth',1);
legend('GT Xentropy','PredictedXentropy','average GT Xentropy','average Predicted Xentropy','Location','northwest')
hold off

gw = 0.9 ; % almost touching, 1 is touching
f1 = bar(samples-gw/6,GTX(:,3),gw/2,'b','EdgeColor','none') ;
xlim([0 101])
ylim([0 1.25])
title('Xentropy for next n=10 pixels')
xlabel('sample') % x-axis label
ylabel('Xentropy') % y-axis label
hold on ;
f2 = bar(samples+gw/6,PRX(:,3),gw/2,'r','EdgeColor','none') ;
hold on
plot(samples,one*mean(GTX(:,3)),'b',samples,one*mean(PRX(:,3)),'r','LineWidth',1);
legend('GT Xentropy','PredictedXentropy','average GT Xentropy','average Predicted Xentropy','Location','northwest')
hold off

gw = 0.9 ; % almost touching, 1 is touching
f1 = bar(samples-gw/6,GTX(:,4),gw/2,'b','EdgeColor','none') ;
xlim([0 101])
ylim([0 1.8])
title('Xentropy for next n=1 pixels')
xlabel('sample') % x-axis label
%ylabel('Xentropy') % y-axis label
hold on ;
f2 = bar(samples+gw/6,PRX(:,4),gw/2,'r','EdgeColor','none') ;
hold on
plot(samples,one*mean(GTX(:,4)),'b',samples,one*mean(PRX(:,4)),'r','LineWidth',1);
legend('GT Xentropy','PredictedXentropy','average GT Xentropy','average Predicted Xentropy','Location','best')
hold off
