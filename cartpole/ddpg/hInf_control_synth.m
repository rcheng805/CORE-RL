%% Code for synthesizing H-infinity controller gains
% Define Model for H-Infinity Controller
g = 9.8;
m = 0.1;
M = 1.1;
l = 0.5;
dt = 0.02;
M = 0.5*M;
l = 0.5*l;
A = [1, dt, 0, 0; 0, 1, -dt*m*g*l/(4*M*l/3 - m*l), 0; 0, 0, 1, dt; ...
    0, 0, M*g*dt/(4*M*l/3 - m*l), 1];
B1 = [0.001; 0.001; 0.001; 0.001];
B2 = [0; dt/M + dt*m*l/(4*M^2*l/3 - m*M*l); 0; -dt/(4*M*l/3 - m*l)];
C1 = eye(4);
C2 = eye(4);
D11 = [1.0; 1.0; 1.0; 1.0];
D12 = [0; 0; 0; 0];
D21 = [0; 0; 0; 0];
D22 = [0; 0; 0; 0];

% Get state space model
P = ss(A,[B1 B2],[C1],[D11 D12], dt);

% Synthesize H-inf controller
[K, CL, gamma] = hinffi(P,1);