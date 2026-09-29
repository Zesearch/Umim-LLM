"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

const mergeRates = [12.1, 21.94, 28.89, 39.53];
const chartTop = 60;
const chartScale = 37;
const chartMax = 86;
const chartMinRate = 10;
const chartMaxRate = 41;

const getXPositions = (width: number) => {
  const left = Math.max(72, width * 0.078);
  const right = width - Math.max(38, width * 0.045);
  return mergeRates.map(
    (rate) => left + ((rate - chartMinRate) / (chartMaxRate - chartMinRate)) * (right - left),
  );
};

const methods = [
  {
    id: "rules",
    label: "MφWT + Rtask",
    setting: "Rule adaptation only",
    summary: "Reuse the WikiText-pretrained merge module; replace only the merge rules.",
    status: "Mφ frozen · LLM frozen",
    result: "71.13% Acc. at 39.53% TR",
    values: [76.95, 75.73, 74.47, 71.13],
  },
  {
    id: "sft",
    label: "MφSFT + Rtask",
    setting: "Task SFT",
    summary: "Initialize from MφWT and fine-tune only the merge module on the target task.",
    status: "Update Mφ only · LLM frozen",
    result: "74.41% Acc. at 39.53% TR",
    values: [78.02, 77.28, 76.31, 74.41],
  },
  {
    id: "adapted",
    label: "Mφtask + Rtask",
    setting: "Task SFT + RL",
    summary: "Continue from MφSFT and optimize only the merge module using DPO sequence pairs.",
    status: "Update Mφ only · LLM frozen",
    result: "81.06% Acc. at 39.53% TR · +2.72 pp over baseline",
    values: [84.1, 84.86, 83.65, 81.06],
  },
] as const;

const yPosition = (value: number) => chartTop + (chartMax - value) * chartScale;

function LineSeries({
  method,
  active,
  positions,
}: {
  method: (typeof methods)[number];
  active: boolean;
  positions: number[];
}) {
  const points = method.values.map((value, index) => ({
    x: positions[index],
    y: yPosition(value),
    value,
  }));

  return (
    <div
      className={`adapted-series adapted-series-${method.id} ${active ? "is-active" : ""}`}
      aria-hidden="true"
    >
      {points.slice(0, -1).map((point, index) => {
        const next = points[index + 1];
        const dx = next.x - point.x;
        const dy = next.y - point.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);

        return (
          <i
            className="adapted-line-segment"
            key={`${method.id}-line-${index}`}
            style={{
              left: `${point.x}px`,
              top: `${point.y}px`,
              width: `${length}px`,
              transform: `rotate(${angle}deg)`,
            }}
          />
        );
      })}

      {points.map((point, index) => {
        const hasCallout = method.id === "adapted" && index === 3;
        return (
          <span
            className={`adapted-data-point ${hasCallout ? "has-callout" : ""}`}
            key={`${method.id}-point-${index}`}
            style={{ left: `${point.x}px`, top: `${point.y}px` }}
          >
            <i />
            <em>{point.value.toFixed(2)}</em>
          </span>
        );
      })}
    </div>
  );
}

export function TaskAdaptedResults() {
  const baseline = 78.34;
  const [activeSeries, setActiveSeries] = useState(0);
  const chartRef = useRef<HTMLDivElement>(null);
  const [chartWidth, setChartWidth] = useState(900);
  const xPositions = getXPositions(chartWidth);

  useLayoutEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const updateWidth = () => setChartWidth(chart.clientWidth);
    updateWidth();

    const observer = new ResizeObserver(updateWidth);
    observer.observe(chart);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActiveSeries((series) => (series + 1) % methods.length);
    }, 4800);

    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="task-adapted-results-section" id="task-adapted-results" aria-labelledby="task-adapted-results-title">
      <div className="task-adapted-results-shell">
        <header className="task-adapted-results-heading">
          <div>
            <p><span>07</span> Adapted Results</p>
            <h2 id="task-adapted-results-title">Merge More. Perform Better.</h2>
          </div>
        </header>

        <figure className="task-adapted-results-figure">
          <figcaption className="adapted-results-caption">
            <div>
              <span>HellaSwag · Llama-3-8B</span>
              <strong>Accuracy across increasing token reduction</strong>
            </div>
            <div className="adapted-bigram-cap">
              <small>Rules re-mined for adaptation</small>
              <strong>Bigram only</strong>
              <span>Theoretical merge ceiling · 50%</span>
            </div>
          </figcaption>

          <div className="adapted-results-body">
            <div className="adapted-chart-panel">
              <div className="adapted-chart-legend" aria-label="Chart legend">
                <span className="adapted-legend-baseline"><i />Baseline</span>
                {methods.map((method, index) => (
                  <button
                    type="button"
                    className={`adapted-legend-${method.id} ${activeSeries === index ? "is-active" : ""}`}
                    key={method.id}
                    onClick={() => setActiveSeries(index)}
                    aria-pressed={activeSeries === index}
                  >
                    <i />{method.label}
                  </button>
                ))}
              </div>

              <div className="adapted-chart-scroll">
                <div ref={chartRef} className="adapted-chart" role="img" aria-label="HellaSwag accuracy at 12.10, 21.94, 28.89, and 39.53 percent token reduction">
                  <span className="adapted-y-title">Accuracy (%) ↑</span>

                  {[70, 75, 80, 85].map((tick) => (
                    <div className="adapted-y-grid" key={tick} style={{ top: `${yPosition(tick)}px` }}>
                      <span>{tick}</span><i />
                    </div>
                  ))}

                  {xPositions.map((x, index) => (
                    <div className="adapted-x-guide" key={mergeRates[index]} style={{ left: `${x}px` }}>
                      <i />
                      <span>{mergeRates[index].toFixed(2)}%</span>
                    </div>
                  ))}

                  <div
                    className="adapted-baseline-line"
                    style={{ top: `${yPosition(baseline)}px` }}
                  >
                    <i />
                    <strong>Uncompressed baseline · 78.34</strong>
                  </div>

                  {methods.map((method, index) => (
                    <LineSeries method={method} active={activeSeries === index} positions={xPositions} key={method.id} />
                  ))}

                  <div
                    className={`adapted-forty-callout ${activeSeries === 2 ? "is-active" : ""}`}
                    style={{
                      left: `${Math.max(76, xPositions[3] - 142)}px`,
                      top: `${yPosition(81.06) - 72}px`,
                    }}
                  >
                    <small>At 39.53% merging</small>
                    <span>+2.72 pp over baseline</span>
                  </div>

                  <strong className="adapted-x-title">Token reduction (%) →</strong>
                </div>
              </div>
            </div>

            <aside className="adapted-result-explanation" aria-label="Differences between task adaptation settings">
              <header>
                <span>What changes?</span>
                <strong>The backbone never does.</strong>
              </header>

              {methods.map((method, index) => (
                <button
                  type="button"
                  className={`adapted-explanation-card adapted-explanation-${method.id} ${activeSeries === index ? "is-active" : ""}`}
                  key={method.id}
                  onClick={() => setActiveSeries(index)}
                  aria-pressed={activeSeries === index}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <h3>{method.label}</h3>
                    <strong>{method.setting}</strong>
                    <p>{method.summary}</p>
                    <small>{method.status}</small>
                    <em>{method.result}</em>
                  </div>
                </button>
              ))}

              <div className="adapted-parameter-note">
                <span>Updated backbone parameters</span>
                <strong>0</strong>
              </div>
            </aside>
          </div>

          <div className="adapted-results-conclusion">
            <span>One lightweight module</span>
            <i aria-hidden="true" />
            <strong>39.53% token reduction</strong>
            <i aria-hidden="true" />
            <strong>+2.72 accuracy points</strong>
          </div>

          <p className="adapted-results-note">
            This adaptation experiment re-mines bigrams only. With non-overlapping token pairs, the theoretical token-reduction ceiling is 50%.
          </p>

          <table className="adapted-results-sr-only">
            <caption>HellaSwag accuracy under task adaptation</caption>
            <thead><tr><th>Method</th>{mergeRates.map((rate) => <th key={rate}>{rate.toFixed(2)}%</th>)}</tr></thead>
            <tbody>
              <tr><th>Baseline</th>{mergeRates.map((rate) => <td key={rate}>{baseline}</td>)}</tr>
              {methods.map((method) => (
                <tr key={method.id}><th>{method.label}</th>{method.values.map((value, index) => <td key={index}>{value}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </figure>
      </div>
    </section>
  );
}
