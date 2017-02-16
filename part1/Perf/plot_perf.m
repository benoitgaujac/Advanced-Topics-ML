formatSpec = '%f%f%f%f%f%f';
perfs_1L    = readtable('Training_OneLinear.csv','Delimiter',';','Format',formatSpec);
perfs_1H    = readtable('Training_OneHidden.csv','Delimiter',';','Format',formatSpec);
perfs_2H    = readtable('Training_TwoHidden.csv','Delimiter',';','Format',formatSpec);
perfs_Conv1 = readtable('Training_Conv1.csv','Delimiter',';','Format',formatSpec);
perfs_conv  = readtable('Training_conv.csv','Delimiter',';','Format',formatSpec);

formatSpec = '%f%f';
test_1L    = readtable('Val_OneLinear.csv','Delimiter',';','Format',formatSpec);
test_1H    = readtable('Val_OneHidden.csv','Delimiter',';','Format',formatSpec);
test_2H    = readtable('Val_TwoHidden.csv','Delimiter',';','Format',formatSpec);
test_Conv1 = readtable('Val_Conv1.csv','Delimiter',';','Format',formatSpec);
test_conv  = readtable('Val_conv.csv','Delimiter',';','Format',formatSpec);

epochs = perfs_2H{:,1};
one = ones(150,1);

f1 = figure
plot(epochs,perfs_1L{1:150,4},'--b',epochs,perfs_1L{1:150,6},'b', epochs,ones(150)*test_1L{1,2},'r','LineWidth',.8)
title('Accuracy OneLinear model')
xlabel('epochs') % x-axis label
ylabel('accuracy') % y-axis label
legend('training accuracy', 'validation accuracy', 'test accuracy','Location','southeast')

f2 = figure
plot(epochs,perfs_1H{1:150,4},'--b',epochs,perfs_1H{1:150,6},'b', epochs,ones(150)*test_1H{1,2},'r','LineWidth',.8)
title('Accuracy OneHidden model')
xlabel('epochs') % x-axis label
ylabel('accuracy') % y-axis label
legend('training accuracy', 'validation accuracy', 'test accuracy','Location','southeast')

f3 = figure
plot(epochs,perfs_2H{1:150,4},'--b',epochs,perfs_2H{1:150,6},'b', epochs,ones(150)*test_2H{1,2},'r','LineWidth',.8)
title('Accuracy TwoHidden model')
xlabel('epochs') % x-axis label
ylabel('accuracy') % y-axis label
legend('training accuracy', 'validation accuracy', 'test accuracy','Location','southeast')

f4 = figure
plot(epochs,perfs_Conv1{1:150,4},'--b',epochs,perfs_Conv1{1:150,6},'b', epochs,ones(150)*test_Conv1{1,2},'r','LineWidth',.8)
title('Accuracy convolutional model')
xlabel('epochs') % x-axis label
ylabel('accuracy') % y-axis label
legend('training accuracy', 'validation accuracy', 'test accuracy','Location','southeast')


col1l = [0 0.6 0];
col1h = [1 0.4 0.4];
col2h = [0.8 0 0];
colConv1 = [0 0 0.8];
colconv = [0.2 0.4 1.0];

f5 = figure
plot(epochs,perfs_Conv1{1:150,4},'--',epochs,perfs_Conv1{1:150,6},'color',colConv1) 
title('Accuracy models')
xlabel('epochs') % x-axis label
ylabel('accuracy') % y-axis label
legend('training accuracy', 'validation accuracy', 'Location','southeast')
hold on;
plot(epochs,perfs_2H{1:150,4},'--',epochs,perfs_2H{1:150,6},'color',col2h) 
legend('training accuracy', 'validation accuracy', 'Location','southeast')
plot(epochs,perfs_1H{1:150,4},'--',epochs,perfs_1H{1:150,6},'color',col1h) 
legend('training accuracy', 'validation accuracy', 'Location','southeast')
plot(epochs,perfs_1L{1:150,4},'--',epochs,perfs_1L{1:150,6},'color',col1l) 
legend('training accuracy', 'validation accuracy', 'Location','southeast')
hold off;