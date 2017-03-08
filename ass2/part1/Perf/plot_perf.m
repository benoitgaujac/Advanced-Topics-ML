formatSpec = '%f%f%f%f%f%f';
perfs_1L    = readtable('Training_gru1l32u.csv','Delimiter',';','Format',formatSpec);
perfs_1H    = readtable('Training_gru1l64u.csv','Delimiter',';','Format',formatSpec);
perfs_2H    = readtable('Training_gru1l128u.csv','Delimiter',';','Format',formatSpec);
perfs_Conv1 = readtable('Training_gru3l32u.csv','Delimiter',';','Format',formatSpec);
%perfs_conv  = readtable('Training_conv.csv','Delimiter',';','Format',formatSpec);

formatSpec = '%f%f';
test_1L    = readtable('test_gru1l32u.csv','Delimiter',';','Format',formatSpec);
test_1H    = readtable('test_gru1l64u.csv','Delimiter',';','Format',formatSpec);
test_2H    = readtable('test_gru1l128u.csv','Delimiter',';','Format',formatSpec);
test_Conv1 = readtable('test_gru3l32u.csv','Delimiter',';','Format',formatSpec);
%test_conv  = readtable('Val_conv.csv','Delimiter',';','Format',formatSpec);

epochs = perfs_1L{2:end,1};
one = ones(99,1);

f1 = figure
plot(epochs,perfs_1L{2:end,4},':b',epochs,perfs_1L{2:end,6},'b', epochs,one*test_1L{1,2},'--b', ...
    epochs,perfs_1L{2:end,3},':r',epochs,perfs_1L{2:end,5},'r', epochs,one*test_1L{1,1},'--r','LineWidth',.8)
title('1 layer 32 units GRU')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.05 1.4])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f2 = figure
plot(epochs,perfs_1H{2:end,4},':b',epochs,perfs_1H{2:end,6},'b', epochs,one*test_1H{1,2},'--b', ...
    epochs,perfs_1H{2:end,3},':r',epochs,perfs_1H{2:end,5},'r', epochs,one*test_1H{1,1},'--r','LineWidth',.8)
title('1 layer 64 units GRU')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
%ylim([0.05 1.4])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f3 = figure
plot(epochs,perfs_2H{2:end,4},':b',epochs,perfs_2H{2:end,6},'b', epochs,one*test_2H{1,2},'--b', ...
    epochs,perfs_2H{2:end,3},':r',epochs,perfs_2H{2:end,5},'r', epochs,one*test_2H{1,1},'--r','LineWidth',.8)
title('1 layer 128 units GRU')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
%ylim([0.05 1.4])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f4 = figure
plot(epochs(1:79),perfs_Conv1{2:end,4},':b',epochs(1:79),perfs_Conv1{2:end,6},'b', epochs(1:79),one(1:79)*test_Conv1{1,2},'--b', ...
    epochs(1:79),perfs_Conv1{2:end,3},':r',epochs(1:79),perfs_Conv1{2:end,5},'r', epochs(1:79),one(1:79)*test_Conv1{1,1},'--r','LineWidth',.8)
title('3 layers 32 units GRU')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.00 1.5])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

col1l = [0 0.6 0];
col1h = [1 0.4 0.4];
col2h = [1 0.6 0.];
colConv1 = [0.2 0.4 1.0];

f5 = figure
plot(epochs,perfs_1L{2:end,3},':',epochs,perfs_1L{2:end,5},'color',col1l,'LineWidth',.8) 
title('Loss models comparison')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.0 1.4])
hold on;
plot(epochs,perfs_1H{2:end,3},':',epochs,perfs_1H{2:end,5},'color',col1h,'LineWidth',.8)
hold on;
plot(epochs,perfs_2H{2:end,3},':',epochs,perfs_2H{2:end,5},'color',col2h,'LineWidth',.8)
hold on;
plot(epochs(1:79),perfs_Conv1{2:end,3},':',epochs(1:79),perfs_Conv1{2:end,5},'color',colConv1,'LineWidth',.8)
hold off;

formatSpec = '%f%f%f%f%f%f';
perfs_132    = readtable('Training_lstm1l32u.csv','Delimiter',';','Format',formatSpec);
perfs_164    = readtable('Training_lstm1l64u.csv','Delimiter',';','Format',formatSpec);
perfs_1128    = readtable('Training_lstm1l128u.csv','Delimiter',';','Format',formatSpec);
perfs_332 = readtable('Training_lstm3l32u.csv','Delimiter',';','Format',formatSpec);
%perfs_conv  = readtable('Training_conv.csv','Delimiter',';','Format',formatSpec);

formatSpec = '%f%f';
test_132    = readtable('test_lstm1l32u.csv','Delimiter',';','Format',formatSpec);
test_164    = readtable('test_lstm1l64u.csv','Delimiter',';','Format',formatSpec);
test_1128    = readtable('test_lstm1l128u.csv','Delimiter',';','Format',formatSpec);
test_332 = readtable('test_lstm3l32u.csv','Delimiter',';','Format',formatSpec);
%test_conv  = readtable('Val_conv.csv','Delimiter',';','Format',formatSpec);

epochs = perfs_132{2:end,1};
one = ones(99,1);

f6 = figure
plot(epochs,perfs_132{2:end,4},':b',epochs,perfs_132{2:end,6},'b', epochs,one*test_132{1,2},'--b', ...
    epochs,perfs_132{2:end,3},':r',epochs,perfs_132{2:end,5},'r', epochs,one*test_132{1,1},'--r','LineWidth',.8)
title('1 layer 32 units LSTM')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.10 2])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f7 = figure
plot(epochs,perfs_164{2:end,4},':b',epochs,perfs_164{2:end,6},'b', epochs,one*test_164{1,2},'--b', ...
    epochs,perfs_164{2:end,3},':r',epochs,perfs_164{2:end,5},'r', epochs,one*test_164{1,1},'--r','LineWidth',.8)
title('1 layer 64 units LSTM')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.10 2])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f8 = figure
plot(epochs,perfs_1128{2:end,4},':b',epochs,perfs_1128{2:end,6},'b', epochs,one*test_1128{1,2},'--b', ...
    epochs,perfs_1128{2:end,3},':r',epochs,perfs_1128{2:end,5},'r', epochs,one*test_1128{1,1},'--r','LineWidth',.8)
title('1 layer 128 units LSTM')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.10 2])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

f9 = figure
plot(epochs,perfs_332{2:end,4},':b',epochs,perfs_332{2:end,6},'b', epochs,one*test_332{1,2},'--b', ...
    epochs,perfs_332{2:end,3},':r',epochs,perfs_332{2:end,5},'r', epochs,one*test_332{1,1},'--r','LineWidth',.8)
title('3 layers 32 units LSTM')
xlabel('epochs') % x-axis label
ylabel('accuracy(%) / loss') % y-axis label
ylim([0.20 2])
legend('training accuracy', 'validation accuracy', 'test accuracy', ...
    'training loss', 'validation loss', 'test loss','Location','northeast')

col1l = [0 0.6 0];
col1h = [1 0.4 0.4];
col2h = [1 0.6 0.];
colConv1 = [0.2 0.4 1.0];

f10 = figure
plot(epochs,perfs_132{2:end,3},':',epochs,perfs_132{2:end,5},'color',col1l,'LineWidth',.8) 
title('Loss models comparison')
xlabel('epochs') % x-axis label
ylabel('loss') % y-axis label
ylim([0.15 2.3])
hold on;
plot(epochs,perfs_164{2:end,3},':',epochs,perfs_164{2:end,5},'color',col1h,'LineWidth',.8)
hold on;
plot(epochs,perfs_1128{2:end,3},':',epochs,perfs_1128{2:end,5},'color',col2h,'LineWidth',.8)
hold on;
plot(epochs,perfs_332{2:end,3},':',epochs,perfs_332{2:end,5},'color',colConv1,'LineWidth',.8)
hold off;