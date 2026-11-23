% =============================================================================
% PlotMCReductionBenchmark.m
% Monte Carlo box+scatter comparison for the exclusive-resource-assignment
% reduction (partial certification). sigma_i = sigma (fixed, alpha = 0.95) for
% every agent; the Monte Carlo noise is instead in the yield costs, drawn iid
% rho_i ~ U(rho_lo, rho_hi) per agent ("heterogeneous" -- no assumed two-group
% structure), n_trials trials per n.
%
% Data file (written by experiments/scalability/run_mc.jl):
%   ../results/mc_rho_noise_results.mat  (wide format: one row per (n, trial), with
%   AC_red/AC_rrf/AC_rrx/AC_full, GI_red/..., t_red/..., search_rrf/rrx,
%   status_red/rrf/rrx/full) -- melted below into the long/tidy (method, n,
%   trial, AC, GI, search, total) format the box_scatter helper expects.
%   Rows whose solver status wasn't LOCALLY_SOLVED/OPTIMAL (Ipopt occasionally
%   reports LOCALLY_INFEASIBLE, or a CC-PNE search finds none) are NaN'd out
%   before plotting -- see "Dropped for non-converged solver status" below.
%
% Figures (saved as PDF into ./figure/):
%   7_mc_cost.pdf        average cost C_avg(z),   box+scatter by method x n
%   8_mc_fairness.pdf    Gini index G(z),         box+scatter by method x n
%   9_mc_searchtime.pdf  CC-PNE search time (RRX, RRF only), log scale
%   10_mc_totaltime.pdf  total time (solid) + search time overlay for RRX/RRF
%                        (dashed), all methods, log scale, 10-second line
% =============================================================================

clear; clc; close all;

% -- Global style (matches PlotReductionBenchmark.m) --------------------------
set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');
set(groot, 'defaultAxesFontSize',    18);
set(groot, 'defaultTextFontSize',    18);
set(groot, 'defaultLegendFontSize',  16);
set(groot, 'defaultAxesLineWidth',   1.2);
set(groot, 'defaultLineLineWidth',   2.0);
set(groot, 'defaultFigureColor',     'w');
set(groot, 'defaultAxesFontName',    'Times New Roman');
set(groot, 'defaultTextFontName',    'Times New Roman');

EXPORT   = true;
% Restricted-CCCE / RR_F times in this file were re-measured by
% experiments/scalability/run_mc.jl (F construction timed + per-n warm-up).
DATAFILE = '../results/mc_rho_noise_results.mat';
FIGDIR   = '../figures';
if EXPORT && ~isfolder(FIGDIR), mkdir(FIGDIR); end

% -- Palette (shared with PlotReductionBenchmark.m / the web report) ----------
C.full    = [0.165 0.471 0.839];     % #2a78d6
C.reduced = [0.922 0.408 0.204];     % #eb6834
C.hullX   = [0.106 0.686 0.478];     % #1baf7a
C.hullF   = [0.929 0.631 0.000];     % #eda100
INK       = [0.106 0.125 0.157];
MUTED     = [0.420 0.459 0.510];
GRID      = [0.843 0.855 0.878];

% Data keys (tbl.method values) vs. the manuscript's formal algorithm names.
METHOD_ORDER = {'Full', 'Reduced', 'Hull / X', 'Hull / F'};
METHOD_LABEL = {'\texttt{Full-CCCE}', '\texttt{Restricted-CCCE}', ...
                '\texttt{RR}${}_{\mathcal{X}}$', '\texttt{RR}${}_{\mathcal{F}}$'};
METHOD_COLOR = [C.full; C.reduced; C.hullX; C.hullF];

% -- Load & melt (wide, one row per n/trial -> long, one row per method) --
D = load(DATAFILE);
tbl = D.tbl;

nRaw   = double(tbl.n(:));
trialG = double(tbl.trial(:));

AC_red_w  = double(tbl.AC_red(:));  AC_rrf_w  = double(tbl.AC_rrf(:));  AC_rrx_w  = double(tbl.AC_rrx(:));  AC_full_w  = double(tbl.AC_full(:));
GI_red_w  = double(tbl.GI_red(:));  GI_rrf_w  = double(tbl.GI_rrf(:));  GI_rrx_w  = double(tbl.GI_rrx(:));  GI_full_w  = double(tbl.GI_full(:));

% Ipopt sometimes reports a non-converged status (e.g. LOCALLY_INFEASIBLE) yet
% JuMP still returns the last iterate's values -- those cost/Gini numbers are
% not trustworthy, so NaN them out here (before the ratio normalization below,
% so a bad Restricted-CCCE baseline correctly poisons that trial's ratios too).
GOOD_STATUS = ["LOCALLY_SOLVED", "OPTIMAL"];
badRed  = ~ismember(string(tbl.status_red(:)),  GOOD_STATUS);
badRrf  = ~ismember(string(tbl.status_rrf(:)),  GOOD_STATUS);
badRrx  = ~ismember(string(tbl.status_rrx(:)),  GOOD_STATUS);
badFull = ~ismember(string(tbl.status_full(:)), GOOD_STATUS) & string(tbl.status_full(:)) ~= "N/A";
AC_red_w(badRed) = NaN;   GI_red_w(badRed) = NaN;
AC_rrf_w(badRrf) = NaN;   GI_rrf_w(badRrf) = NaN;
AC_rrx_w(badRrx) = NaN;   GI_rrx_w(badRrx) = NaN;
AC_full_w(badFull) = NaN; GI_full_w(badFull) = NaN;
fprintf('Dropped for non-converged solver status: Restricted=%d, RRF=%d, RRX=%d, Full=%d (of %d rows)\n', ...
        sum(badRed), sum(badRrf), sum(badRrx), sum(badFull), numel(nRaw));

% Cost/Gini are plotted as-is (not normalized by Restricted-CCCE): the
% earlier normalized view made all four methods collapse to ~1 and hid the
% actual cost/fairness trend across n, which is the point of this figure.
% (The normalized-by-Restricted-CCCE view lives in PlotMCRhoNoise.m's
% figures 12-13, whose whole purpose is showing deviation from that baseline.)

nRep    = numel(nRaw);
method  = [repmat("Reduced", nRep, 1); repmat("Hull / F", nRep, 1); ...
           repmat("Hull / X", nRep, 1); repmat("Full", nRep, 1)];
n       = repmat(nRaw, 4, 1);
trial   = repmat(trialG, 4, 1);
AC      = [AC_red_w; AC_rrf_w; AC_rrx_w; AC_full_w];
GI      = [GI_red_w; GI_rrf_w; GI_rrx_w; GI_full_w];
search  = [zeros(nRep, 1);            double(tbl.search_rrf(:)); double(tbl.search_rrx(:)); zeros(nRep, 1)];
total   = [double(tbl.t_red(:));      double(tbl.t_rrf(:));      double(tbl.t_rrx(:));      double(tbl.t_full(:))];

% NaNs (Full-CCCE past n > full_nmax; EPS_DENOM-excluded cost/Gini trials) are
% left in place and skipped per-group inside box_scatter, since AC/GI/total
% can each have a different NaN pattern now that cost/Gini are normalized.

nList = sort(unique(n));

fprintf('r = %d, c = %.1f, sigma = %.2f (fixed), z_alpha = %.4f (alpha=0.95), Delta = %.1f\n', ...
        D.params.r, D.params.c, D.params.sigma, D.params.zalpha, D.params.delta);
fprintf('rho_i ~ U(%.1f, %.1f) iid per agent (heterogeneous), %d trials/n, n = %s\n', ...
        D.params.rho_lo, D.params.rho_hi, D.n_trials, mat2str(nList'));

% =============================================================================
%% Helper: grouped boxchart + median trend line, one figure
% =============================================================================
% One boxchart call per (n, method) box, each positioned at its own explicit
% x location and given its own explicit color -- this sidesteps the earlier
% boxchart('GroupByColor',...) + swarmchart color-mismatch issue entirely
% (no shared automatic color-group resolution between two different chart
% types), while keeping the native box/whisker/median/outlier look.
function fh = box_scatter(n, y, method, methodOrder, legendLabels, colors, yLabel, titleStr, logScale, ink, muted, gridColor, connectMedians, figSize)
    if nargin < 14 || isempty(figSize), figSize = [820 560]; end
    keep = ismember(method, methodOrder) & ~isnan(y);
    n = n(keep); y = y(keep); method = method(keep);

    nList = sort(unique(n));
    nGroups = numel(methodOrder);
    groupWidth = 0.82;                   % total width spanned by one n's boxes
    boxWidth = groupWidth / nGroups * 0.82 * 0.55;   % thin boxes, matching fig 10's search-time look

    fh = figure('Position', [100 100 figSize(1) figSize(2)]); hold on; box on;
    ax = gca;

    medMat = nan(nGroups, numel(nList));
    xMat   = nan(nGroups, numel(nList));

    for gi = 1:numel(nList)
        for k = 1:nGroups
            sel = (n == nList(gi)) & (method == methodOrder{k});
            v = y(sel);
            if isempty(v), continue; end
            xk  = gi + (k - (nGroups + 1) / 2) * (groupWidth / nGroups);
            col = colors(k, :);
            medMat(k, gi) = median(v);
            xMat(k, gi)   = xk;

            boxchart(ax, xk * ones(numel(v), 1), v, 'BoxWidth', boxWidth, ...
                     'BoxFaceColor', col, 'BoxFaceAlpha', 0.12, 'BoxEdgeColor', col, ...
                     'WhiskerLineColor', col, 'MarkerStyle', 'none', 'LineWidth', 1.2, ...
                     'HandleVisibility', 'off');

            % boxchart's own outlier markers are off (above) -- show every
            % trial as a beeswarm-jittered dot instead, same color as its box.
            swarmchart(ax, xk * ones(numel(v), 1), v, 10, col, 'filled', ...
                       'XJitterWidth', boxWidth * 0.85, 'MarkerFaceAlpha', 0.55, ...
                       'MarkerEdgeAlpha', 0, 'HandleVisibility', 'off');

            plot(xk, median(v), 'd', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', ink, ...
                 'MarkerSize', 5, 'LineWidth', 1, 'HandleVisibility', 'off');
        end
    end

    if connectMedians
        % Median-to-median trend line per method, across n.
        for k = 1:nGroups
            plot(xMat(k, :), medMat(k, :), '--', 'Color', colors(k, :), 'LineWidth', 1.5);
        end
    end

    ax.XTick = 1:numel(nList);
    ax.XTickLabel = string(nList);
    xlim([0.5, numel(nList) + 0.5]);

    if logScale, set(ax, 'YScale', 'log'); end
    set(ax, 'XGrid', 'off', 'YGrid', 'on', 'GridColor', gridColor, 'GridAlpha', 1, 'Layer', 'bottom');

    if ~connectMedians
        % Vertical separators between n-groups instead of a median trend line.
        yl = ylim(ax);
        for gi = 1:(numel(nList) - 1)
            plot([gi + 0.5, gi + 0.5], yl, ':', 'Color', muted, 'LineWidth', 1.1, ...
                 'HandleVisibility', 'off');
        end
        ylim(ax, yl);
    end
    xlabel('Number of agents  n', 'Color', ink);
    ylabel(yLabel, 'Color', ink);
    % (titleStr kept in the signature, unused -- no titles on these figures)

    hLeg = gobjects(nGroups, 1);
    for k = 1:nGroups
        hLeg(k) = patch(NaN, NaN, colors(k, :), 'FaceAlpha', 0.4, 'EdgeColor', colors(k, :));
    end
    legend(hLeg, legendLabels, 'Location', 'best', 'Box', 'on', 'TextColor', ink, 'AutoUpdate', 'off');
    hold off;
end

% =============================================================================
%% FIGURE 7 — Average cost
% =============================================================================
f7 = box_scatter(n, AC, method, METHOD_ORDER, METHOD_LABEL, METHOD_COLOR, ...
                  'Average cost  $C_{avg}(z)$', 'Average cost', false, INK, MUTED, GRID, false, [1200 600]);
if EXPORT, exportgraphics(f7, fullfile(FIGDIR, 'extra_cost.pdf'), 'ContentType', 'vector'); end

% =============================================================================
%% FIGURE 8 — Gini index
% =============================================================================
f8 = box_scatter(n, GI, method, METHOD_ORDER, METHOD_LABEL, METHOD_COLOR, ...
                  'Gini index  $G(z)$', 'Fairness', false, INK, MUTED, GRID, false, [1200 600]);
if EXPORT, exportgraphics(f8, fullfile(FIGDIR, 'extra_fairness.pdf'), 'ContentType', 'vector'); end

% =============================================================================
%% FIGURE 7+8 (test) — cost and fairness side by side, one shared legend
% =============================================================================
fCombo = figure('Position', [100 100 1800 680]);   % taller: shared xlabel + legend row below
tl78 = tiledlayout(fCombo, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Squeezed side by side, so text/boxes need to be bigger than the global
% defaults (18pt axes/text, 16pt legend) to stay readable at this width.
% Tick numbers and axis/legend text are scaled separately since they don't
% need to grow by the same amount.
COMBO_SCALE  = 1.5;
TICK_SCALE   = 1.2;
LABEL_SCALE  = 1.8;
LEGEND_SCALE = 1.5;

ax7c = nexttile(tl78);
draw_box_panel(ax7c, n, AC, method, METHOD_ORDER, METHOD_COLOR, 'Average cost  $C_{avg}(z)$', false, INK, MUTED, GRID, COMBO_SCALE);
ax7c.FontSize = 18 * TICK_SCALE;
ax7c.XLabel.FontSize = 18 * LABEL_SCALE;
ax7c.Title.FontSize = 18 * LABEL_SCALE;

ax8c = nexttile(tl78);
draw_box_panel(ax8c, n, GI, method, METHOD_ORDER, METHOD_COLOR, 'Gini index  $G(z)$', false, INK, MUTED, GRID, COMBO_SCALE);
ax8c.FontSize = 18 * TICK_SCALE;
ax8c.XLabel.FontSize = 18 * LABEL_SCALE;
ax8c.Title.FontSize = 18 * LABEL_SCALE;

hLegC = gobjects(numel(METHOD_ORDER), 1);
for k = 1:numel(METHOD_ORDER)
    hLegC(k) = patch(ax8c, NaN, NaN, METHOD_COLOR(k, :), 'FaceAlpha', 0.4, 'EdgeColor', METHOD_COLOR(k, :));
end
% One shared x-label for both panels (instead of repeating it per tile), and
% the legend as a single horizontal row below the layout so it never covers
% data (reviewer comment: the in-axes legend hid boxes).
xlabel(ax7c, ''); xlabel(ax8c, '');
tl78.XLabel.String      = 'Number of agents  $n$';
tl78.XLabel.Interpreter = 'latex';
tl78.XLabel.FontSize    = 18 * LABEL_SCALE;
tl78.XLabel.Color       = INK;

lgC = legend(ax8c, hLegC, METHOD_LABEL, 'Orientation', 'horizontal', 'NumColumns', numel(METHOD_ORDER), ...
             'Box', 'on', 'TextColor', INK, 'FontSize', 16 * LEGEND_SCALE, 'AutoUpdate', 'off');
lgC.Layout.Tile = 'south';

if EXPORT, exportgraphics(fCombo, fullfile(FIGDIR, 'fig3_cost_fairness.pdf'), 'ContentType', 'vector'); end

% =============================================================================
%% FIGURE 9 — CC-PNE search time (RRX vs. RRF only)
% =============================================================================
SEARCH_ORDER = {'Hull / X', 'Hull / F'};
SEARCH_LABEL = {'\texttt{RR}${}_{\mathcal{X}}$', '\texttt{RR}${}_{\mathcal{F}}$'};
SEARCH_COLOR = [C.hullX; C.hullF];
searchPlot = search; searchPlot(~isnan(searchPlot)) = max(searchPlot(~isnan(searchPlot)), 1e-4);
% (floor for log scale only where a value exists; max(NaN,1e-4) would silently
% turn a missing/zero-PNE search time into a fake floor point otherwise)
f9 = box_scatter(n, searchPlot, method, SEARCH_ORDER, SEARCH_LABEL, SEARCH_COLOR, ...
                  'CC-PNE search time  [s]  (log)', 'PNE search time', true, INK, MUTED, GRID, true);
if EXPORT, exportgraphics(f9, fullfile(FIGDIR, 'extra_search_time.pdf'), 'ContentType', 'vector'); end

% =============================================================================
%% FIGURE 10 — Total time (solid trend) + search time overlay for RRX/RRF
%              (dashed trend, same x position, sits below since search <=
%              total always), with a 10-second reference line
% =============================================================================
% Valid trials only: any trial where ANY method's status wasn't
% LOCALLY_SOLVED/OPTIMAL (i.e. no CC-PNE exists, max_i rho_i < z_alpha*sigma)
% is dropped for ALL methods (total and search), matching the sample set of
% figures 7-8. Set the flag to false to keep every trial in the total time.
badTrial10 = repmat(badRed | badRrf | badRrx | badFull, 4, 1);
% Columns: output file | drop failed trials | add the compact O(n) LP
% (CompactLPHet, experiments/scalability/run_compact.jl) as a fifth method, drawn exactly like
% the others (box + beeswarm + median diamond + solid median trend line).
% Its 600 rows come from the same seed/draws (same 100 instances per n), and
% it gets the same exclusion: its own non-OK status, plus any trial dropped
% for the other methods (these coincide -- 20/600, verified in mc_compact.jl).
FIG10_VARIANTS = {'fig2_total_time.pdf',         true, false; ...
                  'fig2_total_time_compact.pdf', true, true};
COMPACT_FILE  = '../results/mc_compact_results.mat';
COMPACT_LABEL = '\texttt{Compact}';
COMPACT_COLOR = [0 0 0];
badKey10 = nRaw(badRed | badRrf | badRrx | badFull) * 1000 + trialG(badRed | badRrf | badRrx | badFull);
for iv10 = 1:size(FIG10_VARIANTS, 1)
total10 = total; search10 = search; n10 = n; method10 = method;
ORD10 = METHOD_ORDER; COL10 = METHOD_COLOR; LAB10 = METHOD_LABEL;
if FIG10_VARIANTS{iv10, 2}
    total10(badTrial10) = NaN; search10(badTrial10) = NaN;
    fprintf('Fig 10 valid-only: dropped %d trials (per method)\n', sum(badRed | badRrf | badRrx | badFull));
end
if FIG10_VARIANTS{iv10, 3}
    DC  = load(COMPACT_FILE);
    nC  = double(DC.tbl.n(:)); trC = double(DC.tbl.trial(:)); tC = double(DC.tbl.t_cmp(:));
    badC = ~ismember(string(DC.tbl.status_cmp(:)), GOOD_STATUS);
    if FIG10_VARIANTS{iv10, 2}
        badC = badC | ismember(nC * 1000 + trC, badKey10);
    end
    tC(badC) = NaN;
    fprintf('Fig 10 compact: %d rows, dropped %d\n', numel(tC), sum(badC));
    total10  = [total10;  tC];
    search10 = [search10; nan(size(tC))];
    n10      = [n10; nC];
    method10 = [method10; repmat("Compact", numel(tC), 1)];
    ORD10 = [ORD10, {'Compact'}]; COL10 = [COL10; COMPACT_COLOR]; LAB10 = [LAB10, {COMPACT_LABEL}];
end
totalPlot  = total10;  totalPlot(~isnan(totalPlot))   = max(totalPlot(~isnan(totalPlot)), 1e-4);
searchPlot10 = search10; searchPlot10(~isnan(searchPlot10)) = max(searchPlot10(~isnan(searchPlot10)), 1e-4);
SEARCH_METHODS10 = {'Hull / X', 'Hull / F'};  % only these have a nonzero search phase

nGroups10 = numel(ORD10);
groupWidth10 = 0.82;
boxWidth10 = groupWidth10 / nGroups10 * 0.82 * 0.55;   % thin boxes, matching the other figures
LINE_SCALE = 1.1;

f10 = figure('Position', [100 100 1200 600]); hold on; box on;
ax10 = gca;

medTotal10  = nan(nGroups10, numel(nList)); xTotal10  = nan(nGroups10, numel(nList));
medSearch10 = nan(nGroups10, numel(nList)); xSearch10 = nan(nGroups10, numel(nList));

for gi = 1:numel(nList)
    for k = 1:nGroups10
        xk  = gi + (k - (nGroups10 + 1) / 2) * (groupWidth10 / nGroups10);
        col = COL10(k, :);

        selTot = (n10 == nList(gi)) & (method10 == ORD10{k}) & ~isnan(totalPlot);
        vTot = totalPlot(selTot);
        if ~isempty(vTot)
            boxchart(ax10, xk * ones(numel(vTot), 1), vTot, 'BoxWidth', boxWidth10, ...
                     'BoxFaceColor', col, 'BoxFaceAlpha', 0.12, 'BoxEdgeColor', col, ...
                     'WhiskerLineColor', col, 'MarkerStyle', 'none', 'LineWidth', 1.2 * LINE_SCALE, ...
                     'HandleVisibility', 'off');
            swarmchart(ax10, xk * ones(numel(vTot), 1), vTot, 10, col, 'o', 'filled', ...
                       'XJitterWidth', boxWidth10 * 0.85, 'MarkerFaceAlpha', 0.55, ...
                       'MarkerEdgeAlpha', 0, 'HandleVisibility', 'off');
            plot(ax10, xk, median(vTot), 'd', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', INK, ...
                 'MarkerSize', 5, 'LineWidth', 1, 'HandleVisibility', 'off');
            medTotal10(k, gi) = median(vTot); xTotal10(k, gi) = xk;
        end

        if ismember(ORD10{k}, SEARCH_METHODS10)
            selSearch = (n10 == nList(gi)) & (method10 == ORD10{k}) & ~isnan(searchPlot10);
            vSearch = searchPlot10(selSearch);
            if ~isempty(vSearch)
                % boxchart has no dashed-box-edge option, so: draw the box
                % with the SAME fill alpha/color as total (not lighter --
                % that was too faint to read), hide its own solid edge, and
                % overlay a manually-drawn dashed rectangle as the border.
                boxchart(ax10, xk * ones(numel(vSearch), 1), vSearch, 'BoxWidth', boxWidth10, ...
                         'BoxFaceColor', col, 'BoxFaceAlpha', 0.12, 'BoxEdgeColor', 'none', ...
                         'WhiskerLineColor', col, 'MarkerStyle', 'none', 'LineWidth', 1.0 * LINE_SCALE, ...
                         'HandleVisibility', 'off');
                q1 = quantile10(vSearch, 0.25); q3 = quantile10(vSearch, 0.75);
                % rectangle's built-in '--' style is too sparse on a box this
                % small (reads as a few floating ticks, not a box) -- draw a
                % fixed dash count per edge instead so it's dense regardless
                % of box size.
                dashed_rect(ax10, xk - boxWidth10/2, q1, boxWidth10, q3 - q1, col, 1.0 * LINE_SCALE, 8);
                swarmchart(ax10, xk * ones(numel(vSearch), 1), vSearch, 10, col, '^', 'filled', ...
                           'XJitterWidth', boxWidth10 * 0.85, 'MarkerFaceAlpha', 0.55, ...
                           'MarkerEdgeAlpha', 0, 'HandleVisibility', 'off');
                medSearch10(k, gi) = median(vSearch); xSearch10(k, gi) = xk;
            end
        end
    end
end

% Total-time trend: solid. Search-time trend (RRX/RRF only): dashed.
for k = 1:nGroups10
    plot(xTotal10(k, :), medTotal10(k, :), '-', 'Color', COL10(k, :), 'LineWidth', 1.5 * LINE_SCALE);
end
for k = 1:nGroups10
    if ismember(ORD10{k}, SEARCH_METHODS10)
        plot(xSearch10(k, :), medSearch10(k, :), '--', 'Color', COL10(k, :), 'LineWidth', 1.5 * LINE_SCALE);
    end
end

ax10.XTick = 1:numel(nList);
ax10.XTickLabel = string(nList);
xlim([0.5, numel(nList) + 0.5]);
set(ax10, 'YScale', 'log');
set(ax10, 'XGrid', 'off', 'YGrid', 'on', 'GridColor', GRID, 'GridAlpha', 1, 'Layer', 'bottom');

% Vertical separators between n-groups (same style as figures 7-8).
yl10 = ylim(ax10);
for gi = 1:(numel(nList) - 1)
    plot(ax10, [gi + 0.5, gi + 0.5], yl10, ':', 'Color', MUTED, 'LineWidth', 1.1, ...
         'HandleVisibility', 'off');
end
ylim(ax10, yl10);

xlabel('Number of agents  n', 'Color', INK);
ylabel('Computation time  [s]  (log)', 'Color', INK);

yline(10, '--', 'Color', 'k', 'LineWidth', 2.2);

TIME_SCALE = 1.35;
ax10.FontSize = 18 * TIME_SCALE;
ax10.XLabel.FontSize = 18 * TIME_SCALE;
ax10.YLabel.FontSize = 18 * TIME_SCALE;

hLeg10 = gobjects(nGroups10, 1);
for k = 1:nGroups10
    hLeg10(k) = patch(NaN, NaN, COL10(k, :), 'FaceAlpha', 0.4, 'EdgeColor', COL10(k, :));
end
legH10 = hLeg10; legL10 = LAB10; legFS10 = 16 * TIME_SCALE;
if FIG10_VARIANTS{iv10, 3}
    legFS10 = legFS10 * 0.9;   % 5 entries: legend text 10% smaller
end
lgd10 = legend(legH10, legL10, 'Box', 'on', 'TextColor', INK, 'AutoUpdate', 'off', ...
               'FontSize', legFS10);
% 4 legend rows don't fit any corner cleanly against these monotonic trend
% lines on a log axis -- start it outside the axes, then nudge it down so
% it just clips into the empty top margin instead of floating far above.
lgd10.Location = 'northoutside';
lgd10.Orientation = 'horizontal';
lgd10.NumColumns = numel(legH10);
drawnow;
lgdPos10 = lgd10.Position;
lgd10.Position = [lgdPos10(1), lgdPos10(2) + 0.01, lgdPos10(3), lgdPos10(4)];
hold off;
if EXPORT, exportgraphics(f10, fullfile(FIGDIR, FIG10_VARIANTS{iv10, 1}), 'ContentType', 'vector'); end
end   % for iv10 (fig 10 variants)

fprintf('\nSaved figures to %s\n', fullfile(pwd, FIGDIR));

% Draws one box+beeswarm+median-diamond panel into an existing axes, no
% figure/legend of its own -- used to put figures 7 and 8 side by side
% sharing a single legend (see the FIGURE 7+8 combo test above).
function draw_box_panel(ax, n, y, method, methodOrder, colors, yLabel, logScale, ink, muted, gridColor, widthScale)
    if nargin < 12 || isempty(widthScale), widthScale = 1; end
    keep = ismember(method, methodOrder) & ~isnan(y);
    n = n(keep); y = y(keep); method = method(keep);

    nList = sort(unique(n));
    nGroups = numel(methodOrder);
    groupWidth = 0.82;
    boxWidth = groupWidth / nGroups * 0.82 * 0.55 * widthScale;

    hold(ax, 'on'); box(ax, 'on');

    for gi = 1:numel(nList)
        for k = 1:nGroups
            sel = (n == nList(gi)) & (method == methodOrder{k});
            v = y(sel);
            if isempty(v), continue; end
            xk  = gi + (k - (nGroups + 1) / 2) * (groupWidth / nGroups);
            col = colors(k, :);

            boxchart(ax, xk * ones(numel(v), 1), v, 'BoxWidth', boxWidth, ...
                     'BoxFaceColor', col, 'BoxFaceAlpha', 0.12, 'BoxEdgeColor', col, ...
                     'WhiskerLineColor', col, 'MarkerStyle', 'none', 'LineWidth', 1.2 * widthScale, ...
                     'HandleVisibility', 'off');
            swarmchart(ax, xk * ones(numel(v), 1), v, 10 * widthScale, col, 'filled', ...
                       'XJitterWidth', boxWidth * 0.85, 'MarkerFaceAlpha', 0.55, ...
                       'MarkerEdgeAlpha', 0, 'HandleVisibility', 'off');
            plot(ax, xk, median(v), 'd', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', ink, ...
                 'MarkerSize', 5 * widthScale, 'LineWidth', 1, 'HandleVisibility', 'off');
        end
    end

    ax.XTick = 1:numel(nList);
    ax.XTickLabel = string(nList);
    xlim(ax, [0.5, numel(nList) + 0.5]);
    if logScale, set(ax, 'YScale', 'log'); end
    set(ax, 'XGrid', 'off', 'YGrid', 'on', 'GridColor', gridColor, 'GridAlpha', 1, 'Layer', 'bottom');

    yl = ylim(ax);
    for gi = 1:(numel(nList) - 1)
        plot(ax, [gi + 0.5, gi + 0.5], yl, ':', 'Color', muted, 'LineWidth', 1.1, 'HandleVisibility', 'off');
    end
    ylim(ax, yl);

    xlabel(ax, 'Number of agents  n', 'Color', ink);
    title(ax, yLabel, 'Color', ink, 'FontWeight', 'normal');
    hold(ax, 'off');
end

% Toolbox-free quantile (linear interpolation), used only for fig 10's
% manually-drawn dashed search-time box outline.
function q = quantile10(v, p)
    v = sort(v(:));
    idx = (numel(v) - 1) * p + 1;
    lo = floor(idx); hi = ceil(idx);
    if lo == hi
        q = v(lo);
    else
        q = v(lo) + (idx - lo) * (v(hi) - v(lo));
    end
end

% Dashed rectangle with a fixed dash count per edge (rectangle's built-in
% '--' LineStyle is too sparse on a box this small to read as a box).
function dashed_rect(ax, x0, y0, w, h, col, lw, nDashesPerSide)
    corners = [x0, y0; x0 + w, y0; x0 + w, y0 + h; x0, y0 + h; x0, y0];
    nSeg = nDashesPerSide * 2;
    for e = 1:4
        p1 = corners(e, :); p2 = corners(e + 1, :);
        for s = 1:2:nSeg
            t1 = (s - 1) / nSeg; t2 = s / nSeg;
            pa = p1 + t1 * (p2 - p1);
            pb = p1 + t2 * (p2 - p1);
            plot(ax, [pa(1) pb(1)], [pa(2) pb(2)], '-', 'Color', col, ...
                 'LineWidth', lw, 'HandleVisibility', 'off');
        end
    end
end
