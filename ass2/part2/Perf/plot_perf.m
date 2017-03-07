formatSpec = '%f%f%f%f';
perfs_132    = readtable('Training_gru1l32u.csv','Delimiter',';','Format',formatSpec);
perfs_164    = readtable('Training_gru1l64u.csv','Delimiter',';','Format',formatSpec);
perfs_1128    = readtable('Training_gru1l128u.csv','Delimiter',';','Format',formatSpec);
perfs_332 = readtable('Training_gru3l32u.csv','Delimiter',';','Format',formatSpec);

formatSpec = '%f';
test_132    = readtable('test_gru1l32u.csv','Delimiter',';','Format',formatSpec);
test_164    = readtable('test_gru1l64u.csv','Delimiter',';','Format',formatSpec);
test_1128    = readtable('test_gru1l128u.csv','Delimiter',';','Format',formatSpec);
test_332 = readtable('test_gru3l32u.csv','Delimiter',';','Format',formatSpec);

epochs = perfs_132{2:end,1};
one = ones(99,1);

f1 = figure
plot(epochs,perfs_132{2:end,3},':b',epochs,perfs_132{2:end,4},'b', epochs,one*test_132{1,1},'--r','LineWidth',.8)
title('1 layer 32 units GRU')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.08 0.3])
legend('training loss', 'validation loss', 'test loss','Location','northeast')

f2 = figure
plot(epochs,perfs_164{2:end,3},':b',epochs,perfs_164{2:end,4},'b', epochs,one*test_164{1,1},'--r','LineWidth',.8)
title('1 layer 64 units GRU')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.08 0.3])
legend('training loss', 'validation loss', 'test loss','Location','northeast')

f3 = figure
plot(epochs,perfs_1128{2:end,3},':b',epochs,perfs_1128{2:end,4},'b', epochs,one*test_1128{1,1},'--r','LineWidth',.8)
title('1 layer 128 units GRU')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.05 0.3])
legend('training loss', 'validation loss', 'test loss','Location','northeast')

f4 = figure
plot(epochs,perfs_332{2:end,3},':b',epochs,perfs_332{2:end,4},'b', epochs,one*test_332{1,1},'--r','LineWidth',.8)
title('3 layers 32 units GRU')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.05 0.3])
legend('training loss', 'validation loss', 'test loss','Location','northeast')

col1l = [0 0.6 0];
col1h = [1 0.4 0.4];
col2h = [1 0.6 0.];
colConv1 = [0.2 0.4 1.0];

f4 = figure
plot(epochs,perfs_132{2:end,3},':',epochs,perfs_132{2:end,4},'color',col1l,'LineWidth',.8) 
title('Loss models comparison')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.07 0.25])
hold on;
plot(epochs,perfs_164{2:end,3},':',epochs,perfs_164{2:end,4},'color',col1h,'LineWidth',.8)
hold on;
plot(epochs,perfs_1128{2:end,3},':',epochs,perfs_1128{2:end,4},'color',col2h,'LineWidth',.8)
hold on;
plot(epochs,perfs_332{2:end,3},':',epochs,perfs_332{2:end,4},'color',colConv1,'LineWidth',.8)
hold off;