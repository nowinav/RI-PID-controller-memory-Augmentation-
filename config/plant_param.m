% plant_params.m
% Centralized parameters for the buck converter plant model.
% Loaded automatically when plant_model.slx opens (via InitFcn callback).
% Change values here instead of editing block dialogs directly.

%% Source
Vin = 12;          % Input DC voltage (V)

%% Load
R_load = 10;        % Load resistance (Ohm)

%% LC Filter
L = 100e-6;         % Inductance (H)
C = 100e-6;          % Capacitance (F)

%% Switching / PWM
Fsw = 100e3;                 % Switching frequency (Hz) -- 100 kHz
Timer_period = 1/Fsw;        % PWM Generator "Timer period" field (s)
Ts_pwm = 1e-7;                % PWM Generator "Sample time" field (s)

%% Control input (open-loop test only -- replaced by PID/RL later)
D_duty = 0.5;        % Duty cycle command (0 to 1), feeds Gain block -> G

%% Gate drive scaling
% Buck Converter's ideal switch has a Threshold voltage (currently 6 V).
% Gate_gain scales the 0/1 PWM pulse up so it reliably exceeds threshold.
Gate_gain = 10;

%% Solver-related (used by Model Configuration Parameters, not a block)
Solver_name = 'ode23t';
Max_step_size = 1e-6;