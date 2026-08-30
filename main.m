stopTime = simOut.tout(end);
t_V = linspace(0, stopTime, length(simOut.V));

plot(t_V, simOut.V, 'b'); hold on;
plot(t_V, simOut.Vref*ones(size(t_V)), 'r--');
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('V (actual)', 'Vref');
title('V tracking Vref');