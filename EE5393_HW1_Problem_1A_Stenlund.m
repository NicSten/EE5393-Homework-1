function results = estimate_crn_probs(params)
% estimate_crn_probs  Monte Carlo estimation of hitting probabilities for a
% discrete-step stochastic chemical reaction network.
%
% MODEL:
%   State S = [x1,x2,x3] (molecule counts, nonnegative integers)
%   S0 = [110, 26, 55]
%   Reactions:
%     R1: 2X1 + X2 -> 4X3   (k1=1)  Δ1 = [-2,-1,+4]
%     R2: X1 + 2X3 -> 3X2   (k2=2)  Δ2 = [-1,+3,-2]
%     R3: X2 + X3 -> 2X1    (k3=3)  Δ3 = [+2,-1,-1]
%   Weights:
%     w1 = (1/2)*x1*(x1-1)*x2
%     w2 = x1*x3*(x3-1)
%     w3 = 3*x2*x3
%   One reaction fires per step with probabilities proportional to w1,w2,w3.
%
% OUTCOMES (events, tracked as "ever hit" within steps 0..N):
%   C1: x1 >= 150
%   C2: x2 < 10
%   C3: x3 > 100
%
% Usage:
%   results = estimate_crn_probs();                 % defaults
%   results = estimate_crn_probs(struct('M',1e5));  % override M
%
% Output "results" fields:
%   .phat            1x3 estimated hitting probabilities [C1 C2 C3]
%   .ci_normal       3x2 normal-approx 95% CI for each event
%   .ci_wilson       3x2 Wilson 95% CI for each event
%   .earlyW0         number of trajectories terminated due to W=0
%   .steps_mean      mean steps executed per trajectory
%   .running         struct with optional running estimates (if plot enabled)

    % ----------------------- Defaults -----------------------
    if nargin < 1, params = struct(); end
    if ~isfield(params,'S0'), params.S0 = [110, 26, 55]; end
    if ~isfield(params,'N'),  params.N  = 10000; end
    if ~isfield(params,'M'),  params.M  = 50000; end
    if ~isfield(params,'seed'), params.seed = 1; end
    if ~isfield(params,'doPlot'), params.doPlot = true; end
    if ~isfield(params,'plotEvery'), params.plotEvery = 1000; end

    S0   = params.S0(:).';   % row
    N    = params.N;
    M    = params.M;
    seed = params.seed;

    rng(seed, 'twister');

    % Stoichiometry updates (rows correspond to R1,R2,R3)
    dS = [-2, -1, +4;
          -1, +3, -2;
          +2, -1, -1];

    % ----------------------- Bookkeeping -----------------------
    hits = false(M,3);       % hits(m,j)=1 if Cj ever hit in trajectory m
    earlyW0 = 0;
    stepsDone = zeros(M,1);

    % For running estimate plot (store at checkpoints only)
    if params.doPlot
        K = params.plotEvery;
        numPts = floor(M / K);
        run_m = zeros(numPts,1);
        run_phat = zeros(numPts,3);
        pt = 0;
        cumHits = zeros(1,3);
    end

    % ----------------------- Monte Carlo -----------------------
    for m = 1:M
        x1 = S0(1); x2 = S0(2); x3 = S0(3);

        % Check events at step 0 (inclusive horizon 0..N)
        h1 = (x1 >= 150);
        h2 = (x2 < 10);
        h3 = (x3 > 100);

        n = 0;
        while n < N
            if h1 && h2 && h3
                break; % stop early if all three already hit
            end

            % Unnormalized weights
            w1 = 0.5 * x1 * (x1 - 1) * x2;
            w2 = x1 * x3 * (x3 - 1);
            w3 = 3.0 * x2 * x3;
            W  = w1 + w2 + w3;

            if W <= 0
                earlyW0 = earlyW0 + 1;
                break; % stuck trajectory
            end

            % Sample which reaction fires
            u = rand();
            p1 = w1 / W;
            p2 = w2 / W;
            % p3 = 1 - p1 - p2;

            if u < p1
                di = 1;
            elseif u < (p1 + p2)
                di = 2;
            else
                di = 3;
            end

            % Update state
            xn1 = x1 + dS(di,1);
            xn2 = x2 + dS(di,2);
            xn3 = x3 + dS(di,3);

            % Safety guard (should not trigger if weights computed correctly)
            if (xn1 < 0) || (xn2 < 0) || (xn3 < 0)
                % Treat as stuck/invalid and terminate (counts as W=0-like stop)
                earlyW0 = earlyW0 + 1;
                break;
            end

            x1 = xn1; x2 = xn2; x3 = xn3;
            n  = n + 1;

            % Check events after update
            if ~h1 && (x1 >= 150), h1 = true; end
            if ~h2 && (x2 < 10),   h2 = true; end
            if ~h3 && (x3 > 100),  h3 = true; end
        end

        hits(m,:) = [h1, h2, h3];
        stepsDone(m) = n;

        % Running estimate storage
        if params.doPlot
            cumHits = cumHits + double(hits(m,:));
            if mod(m, K) == 0
                pt = pt + 1;
                run_m(pt) = m;
                run_phat(pt,:) = cumHits / m;
            end
        end
    end

    % ----------------------- Estimates + CIs -----------------------
    phat = mean(hits, 1); % 1x3

    z = 1.96;
    ci_normal = zeros(3,2);
    ci_wilson = zeros(3,2);

    for j = 1:3
        p = phat(j);

        % Normal approximation CI
        se = sqrt(max(p*(1-p)/M, 0));
        lo = max(0, p - z*se);
        hi = min(1, p + z*se);
        ci_normal(j,:) = [lo, hi];

        % Wilson score interval
        denom  = 1 + (z^2)/M;
        center = (p + (z^2)/(2*M)) / denom;
        half   = (z * sqrt((p*(1-p))/M + (z^2)/(4*M^2))) / denom;
        ci_wilson(j,:) = [max(0, center-half), min(1, center+half)];
    end

    % ----------------------- Print results -----------------------
    fprintf('Discrete CRN Monte Carlo (M=%d, N=%d, seed=%d)\n', M, N, seed);
    fprintf('Early terminations due to W=0 (or guard): %d (%.4f%%)\n', ...
        earlyW0, 100*earlyW0/M);
    fprintf('Mean steps executed per trajectory: %.2f\n', mean(stepsDone));

    names = {'C1: x1>=150','C2: x2<10','C3: x3>100'};
    for j = 1:3
        fprintf('%s: p_hat=%.6f | 95%% CI normal [%.6f, %.6f] | Wilson [%.6f, %.6f]\n', ...
            names{j}, phat(j), ci_normal(j,1), ci_normal(j,2), ci_wilson(j,1), ci_wilson(j,2));
    end

    % ----------------------- Optional plot -----------------------
    running = struct();
    if params.doPlot
        figure; %#ok<UNRCH>
        plot(run_m, run_phat(:,1), '-'); hold on;
        plot(run_m, run_phat(:,2), '-');
        plot(run_m, run_phat(:,3), '-');
        grid on;
        xlabel('Number of trajectories');
        ylabel('Running estimate of Pr(hit within 0..N)');
        legend({'C1','C2','C3'}, 'Location','best');
        title('Convergence of Monte Carlo hitting-probability estimates');

        running.m = run_m;
        running.phat = run_phat;
    end

    % ----------------------- Pack outputs -----------------------
    results = struct();
    results.phat       = phat;
    results.ci_normal  = ci_normal;
    results.ci_wilson  = ci_wilson;
    results.earlyW0    = earlyW0;
    results.steps_mean = mean(stepsDone);
    results.running    = running;
end
