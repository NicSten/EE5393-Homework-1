function results = partB_crn_meanvar_after7(varargin)
% partB_crn_meanvar_after7
% Exact (dynamic programming) distribution after N steps for the discrete-step CRN,
% then mean/variance of X1, X2, X3. Optional Monte Carlo verification.
%
% Reactions:
%   R1: 2X1 + X2 -> 4X3   k1=1   Δ1=[-2,-1,+4]
%   R2: X1 + 2X3 -> 3X2   k2=2   Δ2=[-1,+3,-2]
%   R3: X2 + X3 -> 2X1    k3=3   Δ3=[+2,-1,-1]
%
% Discrete step weights (propensity-like):
%   w1 = (1/2)*x1*(x1-1)*x2                (k1*C(x1,2)*x2)
%   w2 = x1*x3*(x3-1)                      (k2*x1*C(x3,2) with k2=2)
%   w3 = 3*x2*x3                           (k3*x2*x3 with k3=3)
% One reaction fires per step with p_i = w_i / (w1+w2+w3).
%
% Default: S0=[9,8,7], N=7, doMC=true, M=200000, seed=1.
%
% Usage:
%   results = partB_crn_meanvar_after7();
%   results = partB_crn_meanvar_after7('doMC',false);
%   results = partB_crn_meanvar_after7('M',500000,'seed',42);

% ---------------- Parse inputs ----------------
p = inputParser;
addParameter(p,'S0',[9 8 7], @(v)isnumeric(v)&&numel(v)==3);
addParameter(p,'N',7, @(v)isnumeric(v)&&isscalar(v)&&v>=0);
addParameter(p,'doMC',true, @(v)islogical(v)&&isscalar(v));
addParameter(p,'M',200000, @(v)isnumeric(v)&&isscalar(v)&&v>0);
addParameter(p,'seed',1, @(v)isnumeric(v)&&isscalar(v));
parse(p,varargin{:});

S0   = double(p.Results.S0(:).'); % row
N    = p.Results.N;
doMC = p.Results.doMC;
M    = p.Results.M;
seed = p.Results.seed;

% Stoichiometry updates
dS = [-2, -1, +4;
      -1, +3, -2;
      +2, -1, -1];

% ---------------- Exact DP over states ----------------
% Represent a state as key "x1,x2,x3" in a containers.Map
curr = containers.Map();
curr(stateKey(S0)) = 1.0;

for step = 1:N
    nxt = containers.Map();
    keysCurr = curr.keys;
    for idx = 1:numel(keysCurr)
        k = keysCurr{idx};
        prob = curr(k);
        x = parseKey(k); % [x1 x2 x3]

        [w1,w2,w3] = weights(x);
        W = w1+w2+w3;

        if W <= 0
            % Stuck: treat as absorbing (remains in same state for remaining steps)
            addProb(nxt, k, prob);
            continue;
        end

        ws = [w1 w2 w3];
        for r = 1:3
            if ws(r) <= 0, continue; end
            x2n = x + dS(r,:);
            if any(x2n < 0)
                % Shouldn't happen if weights are correct, but guard anyway
                continue;
            end
            kn = stateKey(x2n);
            addProb(nxt, kn, prob * (ws(r)/W));
        end
    end
    curr = nxt;
end

% Convert final DP map to arrays
finalKeys = curr.keys;
K = numel(finalKeys);
X = zeros(K,3);
P = zeros(K,1);
for i = 1:K
    X(i,:) = parseKey(finalKeys{i});
    P(i) = curr(finalKeys{i});
end

% Normalize (should already sum to 1 up to floating error)
Psum = sum(P);
if Psum > 0
    P = P / Psum;
end

% Exact means/variances (population)
mean_exact = sum(P .* X, 1);
second_exact = sum(P .* (X.^2), 1);
var_exact = second_exact - mean_exact.^2;

% ---------------- Optional Monte Carlo check ----------------
mc = struct();
if doMC
    rng(seed,'twister');
    Xend = zeros(M,3);
    stuckCount = 0;

    for m = 1:M
        x = S0;
        for step = 1:N
            [w1,w2,w3] = weights(x);
            W = w1+w2+w3;
            if W <= 0
                stuckCount = stuckCount + 1;
                break; % absorbing; keep x as-is
            end

            u = rand();
            p1 = w1 / W;
            p2 = w2 / W;

            if u < p1
                r = 1;
            elseif u < (p1 + p2)
                r = 2;
            else
                r = 3;
            end

            xn = x + dS(r,:);
            if any(xn < 0)
                % guard: treat as stuck if something went wrong
                stuckCount = stuckCount + 1;
                break;
            end
            x = xn;
        end
        Xend(m,:) = x;
    end

    mean_mc = mean(Xend, 1);
    var_mc  = var(Xend, 1, 1); % population variance (normalize by M)

    mc.mean = mean_mc;
    mc.var  = var_mc;
    mc.stuck = stuckCount;
    mc.M = M;
    mc.seed = seed;
end

% ---------------- Pack results ----------------
results = struct();
results.S0 = S0;
results.N  = N;
results.exact = struct( ...
    'mean', mean_exact, ...
    'var',  var_exact, ...
    'support_size', K, ...
    'prob_sum', sum(P));
results.states = X;  % optional: exact support states after N steps
results.probs  = P;  % optional: their probabilities
results.mc = mc;

% ---------------- Print summary ----------------
fprintf('Exact DP after N=%d steps from [%d,%d,%d]\n', N, S0(1), S0(2), S0(3));
fprintf('Support size: %d states, total prob = %.12f\n', K, sum(P));
fprintf('Exact mean: [%.6f, %.6f, %.6f]\n', mean_exact(1), mean_exact(2), mean_exact(3));
fprintf('Exact var : [%.6f, %.6f, %.6f]\n', var_exact(1),  var_exact(2),  var_exact(3));

if doMC
    fprintf('\nMonte Carlo check (M=%d, seed=%d):\n', M, seed);
    fprintf('MC mean   : [%.6f, %.6f, %.6f]\n', mc.mean(1), mc.mean(2), mc.mean(3));
    fprintf('MC var    : [%.6f, %.6f, %.6f]\n', mc.var(1),  mc.var(2),  mc.var(3));
    fprintf('Stuck trajectories (W=0 before N): %d (%.4f%%)\n', mc.stuck, 100*mc.stuck/M);
    fprintf('Abs diff (mean): [%.6g, %.6g, %.6g]\n', abs(mc.mean-mean_exact));
    fprintf('Abs diff (var) : [%.6g, %.6g, %.6g]\n', abs(mc.var-var_exact));
end

end

% ===== Helper: compute weights =====
function [w1,w2,w3] = weights(x)
x1 = x(1); x2 = x(2); x3 = x(3);
w1 = 0.5 * x1 * (x1 - 1) * x2; % k1*C(x1,2)*x2 with k1=1
w2 = x1 * x3 * (x3 - 1);       % k2*x1*C(x3,2) with k2=2
w3 = 3.0 * x2 * x3;            % k3*x2*x3 with k3=3
% Ensure no negative weights from weird inputs
if w1 < 0, w1 = 0; end
if w2 < 0, w2 = 0; end
if w3 < 0, w3 = 0; end
end

% ===== Helper: map key <-> state =====
function k = stateKey(x)
k = sprintf('%d,%d,%d', x(1), x(2), x(3));
end

function x = parseKey(k)
parts = sscanf(k, '%d,%d,%d');
x = double(parts(:).');
end

% ===== Helper: accumulate probability into map =====
function addProb(mp, key, val)
if isKey(mp, key)
    mp(key) = mp(key) + val;
else
    mp(key) = val;
end
end