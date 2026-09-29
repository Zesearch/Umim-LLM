"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { MergeModuleStory } from "./merge-module-story";

const clamp = (value: number, min = 0, max = 1) =>
  Math.min(max, Math.max(min, value));

const smoothstep = (start: number, end: number, value: number) => {
  const t = clamp((value - start) / (end - start));
  return t * t * (3 - 2 * t);
};

const toSubscript = (value: string) =>
  value.replace(/[0-9]/g, (digit) => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]);

function CorpusHit({
  children,
  size,
  order,
}: {
  children: string;
  size: 2 | 3 | 4;
  order: number;
}) {
  return (
    <mark
      data-size={size}
      style={{ "--hit-order": order } as CSSProperties}
    >
      {children}
    </mark>
  );
}

const examples = [
  {
    size: 2,
    tokens: ["The", "language", "model", "predicts", "the", "next", "token"],
    mergeStart: 1,
    mergeEnd: 2,
    phrase: "language model",
    mergedIndex: "23",
  },
  {
    size: 4,
    tokens: ["We", "learn", "as", "a", "result", "of", "repetition"],
    mergeStart: 2,
    mergeEnd: 5,
    phrase: "as a result of",
    mergedIndex: "3456",
  },
];

type Example = (typeof examples)[number];

function SequenceToken({
  word,
  index,
  className = "",
  column,
  row,
  mergedIndex,
}: {
  word: string;
  index: number;
  className?: string;
  column: number;
  row: number;
  mergedIndex?: string;
}) {
  return (
    <div
      className={`sequence-token ${className}`}
      style={{ gridColumn: column + 2, gridRow: row }}
    >
      <span>{word}</span>
      <small>x{toSubscript(mergedIndex ?? String(index + 1))}</small>
    </div>
  );
}

function DistributionMatch({ learning }: { learning: number }) {
  const teacher = [23, 55, 86, 39, 18];
  const initial = [68, 25, 43, 73, 31];
  const student = initial.map((height, index) =>
    height + (teacher[index] - height) * learning
  );

  return (
    <div className="distribution-match" aria-label="Predictive distributions converging">
      <div className="distribution-row">
        <span>Teacher</span>
        <div className="probability-bars" aria-hidden="true">
          {teacher.map((height, index) => (
            <i key={index} style={{ height: `${height}%` }} />
          ))}
        </div>
        <small>target distribution</small>
      </div>
      <div className="distribution-row student-distribution">
        <span>Student</span>
        <div className="probability-bars" aria-hidden="true">
          {student.map((height, index) => (
            <i key={index} style={{ height: `${height}%` }} />
          ))}
        </div>
        <small>{learning > 0.82 ? "aligned" : "learning"}</small>
      </div>
    </div>
  );
}

function SequenceExample({
  example,
  progress,
  opacity,
}: {
  example: Example;
  progress: number;
  opacity: number;
}) {
  const span = example.tokens.slice(example.mergeStart, example.mergeEnd + 1);
  const alignColumn = Math.min(
    example.tokens.length - 1,
    example.mergeEnd + Math.floor(progress * (example.tokens.length - example.mergeEnd))
  );
  const alignPercent = ((alignColumn + 0.5) / example.tokens.length) * 100;
  const learning = smoothstep(0.08, 0.72, progress);

  return (
    <div
      className="sequence-example"
      style={{
        opacity,
        transform: `translateY(${(1 - opacity) * 18}px)`,
      }}
      aria-hidden={opacity < 0.35}
    >
      <div className="example-heading">
        <span>{example.size}-token span</span>
        <p>
          <strong>{example.phrase}</strong>
          <em>ends at x{toSubscript(String(example.mergeEnd + 1))}</em>
        </p>
      </div>

      <div className="sequence-grid">
        <div className="track-label" style={{ gridRow: 1 }}>
          <strong>Teacher</strong>
          <span>original sequence</span>
        </div>
        {example.tokens.map((token, index) => (
          <SequenceToken
            key={`teacher-${token}-${index}`}
            word={token}
            index={index}
            column={index}
            row={1}
            className={
              index < example.mergeStart
                ? "unchanged-token"
                : index >= example.mergeStart && index <= example.mergeEnd
                  ? "span-token"
                  : "retained-token"
            }
          />
        ))}

        <div className="track-label merge-label" style={{ gridRow: 2 }}>
          <strong>Trainable Mφ</strong>
          <span>attention pooling</span>
        </div>
        <div
          className="merge-motion"
          style={{
            gridColumn: `${example.mergeStart + 2} / ${example.mergeEnd + 3}`,
            gridRow: 2,
          }}
        >
          <div className="span-embeddings" aria-hidden="true">
            {span.map((token) => <i key={token}>{token}</i>)}
          </div>
          <div className="merge-orb">
            <b>Mφ</b>
            <span>updating</span>
          </div>
          <div className="surrogate-seed">
            ẽ{toSubscript(example.mergedIndex)}
          </div>
        </div>

        <div className="track-label" style={{ gridRow: 3 }}>
          <strong>Student</strong>
          <span>merged sequence</span>
        </div>
        {example.tokens.map((token, index) => {
          if (index >= example.mergeStart && index < example.mergeEnd) {
            return (
              <div
                className="collapsed-position"
                key={`collapsed-${index}`}
                style={{ gridColumn: index + 2, gridRow: 3 }}
              >
                <span />
                <small>collapsed</small>
              </div>
            );
          }
          if (index === example.mergeEnd) {
            return (
              <SequenceToken
                key="surrogate"
                word={example.phrase}
                index={index}
                column={index}
                row={3}
                mergedIndex={example.mergedIndex}
                className="surrogate-token"
              />
            );
          }
          return (
            <SequenceToken
              key={`student-${token}-${index}`}
              word={token}
              index={index}
              column={index}
              row={3}
              className={index < example.mergeStart ? "unchanged-token" : "retained-token"}
            />
          );
        })}

        <div className="token-alignment-area" aria-hidden="true">
          <span
            className="alignment-thread"
            style={{ left: `${alignPercent}%` }}
          >
            <i />
          </span>
        </div>
      </div>

      <div className="distillation-footer">
        <p className="unchanged-note">
          <span />
          Unchanged prefix — no distillation loss
        </p>
        <div className="loss-readout">
          <span>Predictive distillation</span>
          <strong>D<sub>KL</sub>(p<sub>teacher</sub> ∥ p<sub>student</sub>)</strong>
        </div>
        <DistributionMatch learning={learning} />
      </div>
    </div>
  );
}

const distillWords = ["The", "language", "model", "predicts", "the", "next", "token"];

const distillCaptions = [
  "Teacher baseline: the unmerged sequence passes through the frozen Base LLM and produces a next-token distribution at every position.",
  "Student forward: match “language model” in R, merge it with trainable Mφ into a single super token, then run the compressed sequence through the frozen Base LLM.",
  "Align x₂₃ with x₃, distill every changed suffix, and backpropagate through frozen θ to update only Mφ.",
];

const teacherDistributions = [
  [24, 72, 43, 31, 55],
  [58, 32, 78, 46, 20],
  [36, 64, 27, 82, 48],
  [70, 38, 56, 24, 76],
  [42, 80, 33, 62, 26],
  [66, 28, 74, 45, 57],
  [31, 61, 84, 39, 52],
];

const alignmentStudentDistributions = [
  [24, 72, 43, 31, 55],
  null,
  [48, 53, 39, 69, 56],
  [59, 47, 64, 33, 68],
  [51, 69, 42, 55, 36],
  [57, 39, 65, 53, 48],
  [39, 55, 73, 48, 45],
];

const alignmentTeacherLabels = ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆", "x₇"];
const alignmentStudentLabels = ["x₁", "", "x₂₃", "x₄", "x₅", "x₆", "x₇"];

function DemoToken({
  word,
  index,
  className = "",
  column,
  row,
  mergedIndex,
}: {
  word: string;
  index: number;
  className?: string;
  column: number;
  row: number;
  mergedIndex?: string;
}) {
  return (
    <div
      className={`demo-token ${className}`}
      style={{ gridColumn: column, gridRow: row }}
    >
      <span>{word}</span>
      <small>x{toSubscript(mergedIndex ?? String(index + 1))}</small>
    </div>
  );
}

function TeacherNeuralField({
  active,
  flow = "paths",
}: {
  active: boolean;
  flow?: "paths" | "collective";
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let frame = 0;
    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const paths = [
      [1, 4, 7, 9],
      [5, 8, 5, 4],
      [9, 10, 9, 7],
    ];
    const pathColors = ["#c9694d", "#6573c6", "#57949c"];
    const collectiveRoutes = Array.from({ length: 7 }, (_, index) => [
      (index * 3 + 1) % 11,
      (index * 5 + 2) % 13,
      (index * 7 + 4) % 13,
      (index * 4 + 3) % 11,
    ]);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = Math.max(rect.width, 1);
      height = Math.max(rect.height, 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (time = 0) => {
      if (!width || !height) resize();
      context.clearRect(0, 0, width, height);

      const counts = [11, 13, 13, 11];
      const ys = [height * 0.12, height * 0.38, height * 0.64, height * 0.88];
      const pad = Math.max(24, width * 0.055);
      const layers = counts.map((count, layer) =>
        Array.from({ length: count }, (_, index) => ({
          x: pad + (index / (count - 1)) * (width - pad * 2),
          y: ys[layer] + Math.sin(index * 1.7 + layer) * Math.min(5, height * 0.018),
        }))
      );

      for (let layer = 0; layer < layers.length - 1; layer += 1) {
        const fromLayer = layers[layer];
        const toLayer = layers[layer + 1];
        fromLayer.forEach((from, source) => {
          toLayer.forEach((to, target) => {
            const shouldConnect = (source * 7 + target * 11 + layer * 5) % 5 < 2;
            if (!shouldConnect) return;
            context.beginPath();
            context.moveTo(from.x, from.y);
            context.lineTo(to.x, to.y);
            context.strokeStyle = layer === 1
              ? "rgba(83, 104, 174, 0.105)"
              : "rgba(74, 91, 132, 0.085)";
            context.lineWidth = 0.75;
            context.stroke();
          });
        });
      }

      const activeNodes = new Map<string, number>();

      if (flow === "collective") {
        collectiveRoutes.forEach((route, routeIndex) => {
          const progress = reducedMotion
            ? ((routeIndex + 0.45) / collectiveRoutes.length) % 1
            : (time / 10000 + routeIndex / collectiveRoutes.length) % 1;
          const segmentPosition = progress * (layers.length - 1);
          const layer = Math.min(layers.length - 2, Math.floor(segmentPosition));
          const localProgress = segmentPosition - layer;
          const from = layers[layer][route[layer]];
          const to = layers[layer + 1][route[layer + 1]];
          const trailStart = Math.max(0, localProgress - 0.14);
          const startX = from.x + (to.x - from.x) * trailStart;
          const startY = from.y + (to.y - from.y) * trailStart;
          const signalX = from.x + (to.x - from.x) * localProgress;
          const signalY = from.y + (to.y - from.y) * localProgress;
          const tone = routeIndex % 3 === 0 ? "112, 143, 148" : "116, 128, 157";

          context.beginPath();
          context.moveTo(startX, startY);
          context.lineTo(signalX, signalY);
          context.strokeStyle = `rgba(${tone}, 0.32)`;
          context.lineWidth = 1.15;
          context.lineCap = "round";
          context.shadowColor = `rgba(${tone}, 0.28)`;
          context.shadowBlur = 4;
          context.stroke();
          context.shadowBlur = 0;

          const sourceKey = `${layer}-${route[layer]}`;
          const targetKey = `${layer + 1}-${route[layer + 1]}`;
          activeNodes.set(sourceKey, Math.max(activeNodes.get(sourceKey) ?? 0, 0.4 * (1 - localProgress)));
          activeNodes.set(targetKey, Math.max(activeNodes.get(targetKey) ?? 0, 0.4 * localProgress));
        });
      } else {
        paths.forEach((path, pathIndex) => {
          for (let layer = 0; layer < path.length - 1; layer += 1) {
            const from = layers[layer][path[layer]];
            const to = layers[layer + 1][path[layer + 1]];
            const color = pathColors[pathIndex];
            context.beginPath();
            context.moveTo(from.x, from.y);
            context.lineTo(to.x, to.y);
            context.strokeStyle = `${color}b8`;
            context.lineWidth = 1.6;
            context.stroke();

            const phase = reducedMotion ? 0.55 : (time / 1700 + layer * 0.19 + pathIndex * 0.27) % 1;
            const signalX = from.x + (to.x - from.x) * phase;
            const signalY = from.y + (to.y - from.y) * phase;
            context.beginPath();
            context.arc(signalX, signalY, 2.8, 0, Math.PI * 2);
            context.fillStyle = color;
            context.shadowColor = color;
            context.shadowBlur = 10;
            context.fill();
            context.shadowBlur = 0;
          }
        });
      }

      layers.forEach((layer, layerIndex) => {
        layer.forEach((node, nodeIndex) => {
          const pathIndex = paths.findIndex((path) => path[layerIndex] === nodeIndex);
          const collectiveIntensity = activeNodes.get(`${layerIndex}-${nodeIndex}`) ?? 0;
          const highlighted = flow === "collective" ? collectiveIntensity > 0.05 : pathIndex >= 0;
          const nodeScale = Math.max(0.62, Math.min(1, height / 140));
          const radius = (layerIndex === 0 || layerIndex === layers.length - 1 ? 6.2 : 4.7) * nodeScale;
          context.beginPath();
          context.arc(node.x, node.y, radius, 0, Math.PI * 2);
          if (flow === "collective") {
            context.fillStyle = highlighted
              ? `rgba(221, 232, 238, ${0.92 + collectiveIntensity * 0.08})`
              : "rgba(248, 250, 254, 0.98)";
            context.strokeStyle = highlighted
              ? `rgba(103, 127, 145, ${0.58 + collectiveIntensity * 0.32})`
              : "rgba(77, 88, 119, 0.48)";
            context.lineWidth = highlighted ? 1.35 : 1.05;
            if (highlighted) {
              context.shadowColor = "rgba(103, 133, 145, 0.48)";
              context.shadowBlur = 5 + collectiveIntensity * 8;
            }
          } else {
            context.fillStyle = highlighted ? pathColors[pathIndex] : "rgba(248, 250, 254, 0.98)";
            context.strokeStyle = highlighted ? `${pathColors[pathIndex]}d9` : "rgba(77, 88, 119, 0.48)";
            context.lineWidth = highlighted ? 1.7 : 1.05;
            if (highlighted) {
              context.shadowColor = pathColors[pathIndex];
              context.shadowBlur = 11;
            }
          }
          context.fill();
          context.stroke();
          context.shadowBlur = 0;
        });
      });
    };

    const loop = (time: number) => {
      draw(time);
      frame = window.requestAnimationFrame(loop);
    };
    const observer = new ResizeObserver(() => {
      resize();
      draw(performance.now());
    });

    observer.observe(canvas);
    resize();
    if (active && !reducedMotion) frame = window.requestAnimationFrame(loop);
    else draw(performance.now());

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [active, flow]);

  return (
    <div className="teacher-neural-field">
      <canvas
        ref={canvasRef}
        className="teacher-neural-canvas"
        role="img"
        aria-label="Dense frozen neural network with input, hidden, and output units"
      />
    </div>
  );
}

function DistillationAnimation({
  step,
  paused,
  onSelectStep,
  onResume,
}: {
  step: number;
  paused: boolean;
  onSelectStep: (step: number) => void;
  onResume: () => void;
}) {
  return (
    <div className={`distill-demo distill-step-${step}`}>
      <div className="demo-legend">
        <span className="demo-rule-state"><i /> Merge Rules <b>R</b></span>
        <span className="demo-frozen-state"><i /> Base LLM θ <b>Shared · Frozen</b></span>
        <span className="demo-trainable-state"><i /> Merge Module Mφ <b>Trainable</b></span>
      </div>

      <div className="demo-canvas">
        <div className="teacher-baseline" aria-label="Teacher model forward pass without merging">
          <div className="teacher-baseline-label teacher-input-label" style={{ gridColumn: 1, gridRow: 1 }}>
            <strong>Teacher input</strong>
          </div>
          {distillWords.map((word, index) => (
            <DemoToken
              key={`teacher-input-${word}-${index}`}
              word={word}
              index={index}
              column={index + 2}
              row={1}
              className="teacher-input-token"
            />
          ))}
          {distillWords.map((_, index) => (
            <span
              key={`teacher-input-flow-${index}`}
              className="teacher-input-flow"
              style={{ gridColumn: index + 2, gridRow: 2, "--flow-order": index } as CSSProperties}
              aria-hidden="true"
            ><i /></span>
          ))}

          <div className="teacher-network" style={{ gridColumn: "1 / 9", gridRow: 3 }}>
            <div className="teacher-network-heading">
              <strong>Frozen Base LLM <b>θ</b></strong>
            </div>
            <div className="teacher-network-core" aria-hidden="true">
              <TeacherNeuralField active={step === 0} flow="collective" />
            </div>
          </div>

          {distillWords.map((_, index) => (
            <span
              key={`teacher-output-flow-${index}`}
              className="teacher-output-flow"
              style={{ gridColumn: index + 2, gridRow: 4, "--flow-order": index } as CSSProperties}
              aria-hidden="true"
            ><i /></span>
          ))}
          <div className="teacher-baseline-label teacher-output-label" style={{ gridColumn: 1, gridRow: 5 }}>
            <strong>Teacher outputs</strong>
          </div>
          {teacherDistributions.map((distribution, index) => (
            <div
              className="teacher-probability"
              key={`teacher-probability-${index}`}
              style={{ gridColumn: index + 2, gridRow: 5 }}
            >
              <small>p<sub>θ</sub>(·|x<sub>≤{index + 1}</sub>)</small>
              <div aria-hidden="true">
                {distribution.map((height, bar) => <i key={bar} style={{ height: `${height}%` }} />)}
              </div>
            </div>
          ))}
        </div>

        <div className="student-merge-forward" aria-label="Student forward pass with a merged super token">
          <div className="student-forward-label" style={{ gridColumn: 1, gridRow: 1 }}>
            <strong>Original sequence</strong>
          </div>
          {distillWords.map((word, index) => (
            <DemoToken
              key={`student-source-${word}-${index}`}
              word={word}
              index={index}
              column={index + 2}
              row={1}
              className={index === 1 || index === 2 ? "student-source-rule" : ""}
            />
          ))}
          <div className="student-rule-match" style={{ gridColumn: "3 / 5", gridRow: 1 }}>
            <span>2-gram in Merge Rules R</span>
          </div>

          <div className="student-merge-bridge" style={{ gridColumn: 4, gridRow: 2 }}>
            <span className="student-merge-path student-merge-path-in" aria-hidden="true" />
            <div className="student-merge-module">
              <strong>Mφ</strong>
              <small>Merge module</small>
            </div>
            <span className="student-merge-path student-merge-path-out" aria-hidden="true" />
          </div>

          <div className="student-forward-label" style={{ gridColumn: 1, gridRow: 3 }}>
            <strong>Student sequence</strong>
          </div>
          <DemoToken word="The" index={0} column={2} row={3} />
          <div className="student-empty-position" style={{ gridColumn: 3, gridRow: 3 }}>
            <i />
          </div>
          <DemoToken
            word="language model"
            index={2}
            column={4}
            row={3}
            mergedIndex="23"
            className="student-forward-super"
          />
          {distillWords.slice(3).map((word, offset) => (
            <DemoToken
              key={`student-forward-token-${word}-${offset}`}
              word={word}
              index={offset + 3}
              column={offset + 5}
              row={3}
            />
          ))}

          {[2, 4, 5, 6, 7, 8].map((column, index) => (
            <span
              key={`student-model-input-${column}`}
              className="student-model-flow"
              style={{ gridColumn: column, gridRow: 4, "--flow-order": index } as CSSProperties}
              aria-hidden="true"
            ><i /></span>
          ))}
          <div className="student-forward-label student-model-label" style={{ gridColumn: 1, gridRow: 5 }}>
            <strong>Student model</strong>
          </div>
          <div className="student-frozen-network" style={{ gridColumn: "2 / 9", gridRow: 5 }}>
            <TeacherNeuralField active={step === 1} flow="collective" />
          </div>
          {[2, 4, 5, 6, 7, 8].map((column, index) => (
            <span
              key={`student-model-output-${column}`}
              className="student-model-flow student-output-flow"
              style={{ gridColumn: column, gridRow: 6, "--flow-order": index } as CSSProperties}
              aria-hidden="true"
            ><i /></span>
          ))}

          <div className="student-forward-label student-output-label" style={{ gridColumn: 1, gridRow: 7 }}>
            <strong>Student outputs</strong>
          </div>
          {[0, null, 2, 3, 4, 5, 6].map((distributionIndex, slot) => (
            distributionIndex === null ? (
              <div className="student-no-output" key="student-no-output" style={{ gridColumn: slot + 2, gridRow: 7 }}>
                <i />
              </div>
            ) : (
              <div
                className="teacher-probability student-probability"
                key={`student-probability-${slot}`}
                style={{ gridColumn: slot + 2, gridRow: 7 }}
              >
                <small>p<sub>θ</sub>(·|x<sub>{slot === 2 ? "₂₃" : `≤${slot + 1}`}</sub>)</small>
                <div aria-hidden="true">
                  {teacherDistributions[distributionIndex].map((height, bar) => (
                    <i key={bar} style={{ height: `${height}%` }} />
                  ))}
                </div>
              </div>
            )
          ))}
        </div>

        <div className="distillation-alignment" aria-label="Teacher and student predictive distributions aligned for distillation">
          <div className="alignment-scope" style={{ gridColumn: "2 / 9", gridRow: 1 }}>
            <span>Soft-target alignment</span>
            <strong>Distill only the computation changed by merging</strong>
            <span>x₃ ↔ x₂₃ · changed suffix</span>
          </div>

          <div className="alignment-row-label alignment-teacher-label" style={{ gridColumn: 1, gridRow: 2 }}>
            <span>Teacher targets</span>
            <strong>Unmerged</strong>
            <small>Base LLM θ · frozen</small>
          </div>
          {teacherDistributions.map((distribution, slot) => (
            <div
              className={`alignment-probability alignment-teacher-probability ${slot === 0 ? "alignment-unchanged-card" : ""} ${slot === 1 ? "alignment-absorbed-source" : ""} ${slot === 2 ? "alignment-anchor-card" : ""}`}
              key={`alignment-teacher-${slot}`}
              style={{ gridColumn: slot + 2, gridRow: 2 }}
            >
              <div className="alignment-token-label">
                <b>{alignmentTeacherLabels[slot]}</b>
                <span>{distillWords[slot]}</span>
              </div>
              <div className="alignment-bars" aria-hidden="true">
                {distribution.map((height, bar) => <i key={bar} style={{ height: `${height}%` }} />)}
              </div>
              <small>p<sup>T</sup><sub>{slot + 1}</sub></small>
            </div>
          ))}

          <div className="alignment-row-label alignment-loss-label" style={{ gridColumn: 1, gridRow: 3 }}>
            <span>Alignment</span>
            <strong>Soft targets</strong>
            <small>D<sub>KL</sub>(p<sup>T</sup> ∥ p<sup>S</sup>)</small>
          </div>
          <div className="alignment-no-loss" style={{ gridColumn: 2, gridRow: 3 }}>
            <i>=</i><strong>No loss</strong><small>unchanged</small>
          </div>
          <div className="alignment-absorbed" style={{ gridColumn: 3, gridRow: 3 }}>
            <i /><strong>x₂ absorbed</strong><small>into x₂₃</small>
          </div>
          {[2, 3, 4, 5, 6].map((slot, order) => (
            <div
              className={`alignment-zip-link ${order === 0 ? "alignment-anchor-link" : "alignment-suffix-link"}`}
              key={`alignment-link-${slot}`}
              style={{
                gridColumn: slot + 2,
                gridRow: 3,
                "--alignment-order": order,
              } as CSSProperties}
            >
              <span>{order === 0 ? "ALIGN" : "KL"}</span>
              <i />
            </div>
          ))}

          <div className="alignment-row-label alignment-student-label" style={{ gridColumn: 1, gridRow: 4 }}>
            <span>Student predictions</span>
            <strong>Merged</strong>
            <small>Mφ trainable · θ frozen</small>
          </div>
          {alignmentStudentDistributions.map((distribution, slot) => (
            distribution === null ? (
              <div className="alignment-empty-card" key="alignment-empty" style={{ gridColumn: slot + 2, gridRow: 4 }}>
                <i /><strong>absorbed</strong><small>no student state</small>
              </div>
            ) : (
              <div
                className={`alignment-probability alignment-student-probability ${slot === 0 ? "alignment-unchanged-card" : ""} ${slot === 2 ? "alignment-anchor-card" : ""}`}
                key={`alignment-student-${slot}`}
                style={{ gridColumn: slot + 2, gridRow: 4, "--alignment-order": Math.max(0, slot - 2) } as CSSProperties}
              >
                <div className="alignment-token-label">
                  <b>{alignmentStudentLabels[slot]}</b>
                  <span>{slot === 2 ? "language model" : distillWords[slot]}</span>
                </div>
                <div className="alignment-bars" aria-hidden="true">
                  {distribution.map((height, bar) => <i key={bar} style={{ height: `${height}%` }} />)}
                </div>
                <small>p<sup>S</sup><sub>{slot === 2 ? "23" : slot + 1}</sub></small>
              </div>
            )
          ))}

          <div className="alignment-training-closure" style={{ gridColumn: "2 / 9", gridRow: 5 }}>
            <div className="alignment-loss-node">
              <small>Σ aligned KL</small>
              <strong>ℒ<sub>distill</sub></strong>
            </div>
            <div className="alignment-gradient-path">
              <i><b /></i>
              <span>Gradients flow through frozen θ</span>
              <small>θ receives no parameter update</small>
            </div>
            <div className="alignment-update-node">
              <small>Update only</small>
              <strong>Mφ</strong>
              <em>Trainable</em>
            </div>
          </div>
        </div>

        <div className="demo-single-sequence">
          <div className="demo-match-evidence">
            <span>2-gram rule in R</span>
            <strong>language model</strong>
            <em>(x₂, x₃) ∈ R</em>
            <i aria-hidden="true" />
          </div>
          <div className="demo-single-row">
            <div className="demo-row-label">
              <strong>Sequence</strong>
              <span>token embeddings</span>
            </div>
            {distillWords.map((word, index) => (
              <DemoToken
                key={`single-${word}-${index}`}
                word={word}
                index={index}
                column={index + 2}
                row={1}
                className={index === 1 || index === 2 ? "merge-source" : ""}
              />
            ))}
            <div className="demo-rule-outline" aria-hidden="true">
              <span>match in R</span>
            </div>
            <DemoToken
              word="language model"
              index={2}
              column={4}
              row={1}
              mergedIndex="23"
              className="working-surrogate"
            />
          </div>

          <div className="demo-merge-operation" aria-hidden="true">
            <div className="merge-inputs"><span>e₂</span><span>e₃</span></div>
            <i />
            <div className="demo-merge-module"><strong>Mφ</strong><small>attention pooling</small></div>
            <i />
            <div className="merge-output">ẽ₂₃</div>
          </div>
        </div>

        <div className="demo-paired-sequences">
          <div className="demo-row-label teacher-label" style={{ gridColumn: 1, gridRow: 1 }}>
            <strong>Teacher</strong>
            <span>original sequence</span>
          </div>
          {distillWords.map((word, index) => (
            <DemoToken
              key={`teacher-demo-${word}-${index}`}
              word={word}
              index={index}
              column={index + 2}
              row={1}
              className={index === 2 ? "teacher-anchor" : ""}
            />
          ))}

          <div className="demo-row-label student-label" style={{ gridColumn: 1, gridRow: 3 }}>
            <strong>Student</strong>
            <span>merged sequence</span>
          </div>
          <DemoToken word="The" index={0} column={2} row={3} className="unchanged-demo-token" />
          <div className="demo-collapsed-slot" style={{ gridColumn: 3, gridRow: 3 }}>
            <i /><small>x₂ collapsed</small>
          </div>
          <DemoToken
            word="language model"
            index={2}
            column={4}
            row={3}
            mergedIndex="23"
            className="student-surrogate"
          />
          {distillWords.slice(3).map((word, offset) => (
            <DemoToken
              key={`student-demo-${word}-${offset}`}
              word={word}
              index={offset + 3}
              column={offset + 5}
              row={3}
              className="student-suffix-token"
            />
          ))}

          {[4, 5, 6, 7, 8].map((column, index) => (
            <span
              key={column}
              className={`demo-distill-link ${index === 0 ? "anchor-link" : "suffix-link"}`}
              style={{ gridColumn: column, gridRow: 2, "--link-order": index } as CSSProperties}
            >
              <i />
              {index === 0 && <small>x₃ ↕ x₂₃</small>}
            </span>
          ))}

          <div className="demo-prefix-note" style={{ gridColumn: "2 / 4", gridRow: 4 }}>
            Unchanged prefix
          </div>
          <div className="demo-module-note" style={{ gridColumn: "3 / 5", gridRow: 4 }}>
            <i /> Mφ produces x₂₃
          </div>
          <div className="demo-suffix-bracket" style={{ gridColumn: "4 / 9", gridRow: 4 }}>
            <span>Distill aligned position + changed suffix</span>
          </div>

          <div className="demo-shared-model" style={{ gridColumn: 9, gridRow: "1 / 4" }}>
            <span>same weights</span>
            <strong>Base LLM</strong>
            <b>θ</b>
            <em>Frozen</em>
            <i className="teacher-port" />
            <i className="student-port" />
          </div>
        </div>
      </div>

      <div className="demo-explanation" aria-live="polite">
        <div className="demo-step-meter" aria-label="Select and pause an animation frame">
          {distillCaptions.map((caption, index) => (
            <button
              key={caption}
              type="button"
              className={index === step ? "active" : index < step ? "complete" : ""}
              aria-label={`Show frame ${index + 1}: ${caption}`}
              aria-pressed={paused && index === step}
              title={`Frame ${index + 1}`}
              onClick={() => onSelectStep(index)}
            />
          ))}
          {paused && (
            <button type="button" className="demo-step-resume" onClick={onResume}>
              Play
            </button>
          )}
        </div>
        <p><span>0{step + 1}</span>{distillCaptions[step]}</p>
        <strong>D<sub>KL</sub>(p<sub>teacher</sub> ∥ p<sub>student</sub>)</strong>
      </div>
    </div>
  );
}

const prefillWords = ["The", "language", "model", "predicts", "the", "next", "token", "in", "the", "world"];
const compressedPrefillTokens = [
  { word: "The", index: "1", merged: false },
  { word: "language model", index: "23", merged: true },
  { word: "predicts", index: "4", merged: false },
  { word: "the next token", index: "5:7", merged: true },
  { word: "in", index: "8", merged: false },
  { word: "the", index: "9", merged: false },
  { word: "world", index: "10", merged: false },
];

const inferenceCaptions = [
  "Merge matched prompt spans before the first LLM forward, then build the initial compressed KV cache and first next-token distribution.",
  "Prefill predicts “is”. After appending its token ID, UMIM checks the new tail. No merge rule matches, so “is” is forwarded normally and adds one new KV entry.",
  "Decoding predicts “a”. The tail “is a” matches a 2-gram rule, so UMIM rolls back the KV for “is” and replaces the pair with one merged KV state.",
];

const decodeDistribution = [34, 68, 47, 88, 55, 29];

function PrefillAnimation({
  active,
  step,
  paused,
  onSelectStep,
  onResume,
}: {
  active: boolean;
  step: number;
  paused: boolean;
  onSelectStep: (step: number) => void;
  onResume: () => void;
}) {
  const decodedTokens = [...compressedPrefillTokens, { word: "is", index: "11", merged: false }];

  return (
    <div className={`prefill-demo inference-step-${step} ${active ? "prefill-active" : ""}`}>
      <div className="inference-legend">
        <span className="inference-rule"><i /> Merge Rules <b>R · non-overlapping</b></span>
        <span className="inference-module"><i /> Merge Module <b>Mφ · trained</b></span>
        <span className="inference-frozen"><i /> Base LLM <b>θ · frozen</b></span>
      </div>

      <div className="prefill-canvas">
        <div className="prefill-original-row">
          <div className="prefill-row-label prefill-simple-label"><strong>Prompt</strong></div>
          <div className="prefill-original-strip">
            {prefillWords.map((word, index) => (
              <div
                className="prefill-token"
                key={`prefill-original-${word}-${index}`}
                style={{ gridColumn: index + 1, gridRow: 1 }}
              >
                <span>{word}</span><small>x{toSubscript(String(index + 1))}</small>
              </div>
            ))}
            <div className="prefill-rule-outline prefill-rule-two" aria-hidden="true"><span>2-gram in R</span></div>
            <div className="prefill-rule-outline prefill-rule-three" aria-hidden="true"><span>3-gram in R</span></div>
          </div>
        </div>

        <div className="prefill-merge-row" aria-label="Matched spans compressed by the trained merge module">
          <div className="prefill-row-label prefill-simple-label"><strong>Apply Mφ</strong></div>
          <div className="prefill-merge-grid">
            <div className="prefill-merge-operation prefill-merge-two">
              <span className="prefill-merge-path prefill-merge-path-in" aria-hidden="true" />
              <strong><b>Mφ</b></strong>
              <span className="prefill-merge-path prefill-merge-path-out" aria-hidden="true" />
            </div>
            <div className="prefill-merge-operation prefill-merge-three">
              <span className="prefill-merge-path prefill-merge-path-in" aria-hidden="true" />
              <strong><b>Mφ</b></strong>
              <span className="prefill-merge-path prefill-merge-path-out" aria-hidden="true" />
            </div>
          </div>
        </div>

        <div className="prefill-compressed-row">
          <div className="prefill-row-label prefill-simple-label"><strong>Compressed Sequence</strong></div>
          <div className="prefill-compressed-strip">
            {compressedPrefillTokens.map((token) => (
              <div className={`prefill-token ${token.merged ? "prefill-super-token" : ""}`} key={token.index}>
                <span>{token.word}</span><small>x{toSubscript(token.index)}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="prefill-model-row">
          <div className="prefill-row-label prefill-simple-label"><strong>Base LLM Forward</strong></div>
          <div className="prefill-frozen-network">
            <TeacherNeuralField active={active && step === 0} flow="collective" />
          </div>
        </div>

        <div className="prefill-results-row">
          <div className="prefill-row-label prefill-simple-label"><strong>Prefill Output</strong></div>
          <div className="prefill-results">
            <div className="prefill-kv-panel">
              <div className="prefill-panel-heading">
                <div><span>Initial KV Cache</span></div>
                <b>baseline 10 → UMIM 7</b>
              </div>
              <div className="prefill-kv-strip" aria-label="Seven compressed key-value cache entries">
                {compressedPrefillTokens.map((token, index) => (
                  <div className={`prefill-kv-slot ${token.merged ? "prefill-super-kv" : ""}`} key={`kv-${token.index}`} style={{ "--kv-order": index } as CSSProperties}>
                    <span><i>K</i><i>V</i></span>
                  </div>
                ))}
              </div>
            </div>

            <div className="prefill-next-token">
              <div className="prefill-panel-heading">
                <div><strong>First Next-token Distribution</strong></div>
              </div>
              <div className="prefill-next-body">
                <div className="prefill-next-bars" aria-hidden="true">
                  {[38, 74, 46, 92, 57, 31].map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="decode-frame decode-normal-frame" aria-label="Normal autoregressive decoding when the generated tail does not match a merge rule">
        <div className="decode-frame-heading">
          <div><span>Decoding</span></div>
          <div className="decode-counts"><span>Effective positions <b>7 → 8</b></span><span>KV length <b>7 → 8</b></span></div>
        </div>

        <div className="decode-normal-state">
          <div className="decode-side-label decode-simple-label"><strong>Sample &amp; Append</strong></div>
          <div className="decode-prefill-sample">
            <strong>Next-token Prediction</strong>
            <div>{decodeDistribution.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</div>
            <span>sample</span>
            <b>“is”</b>
          </div>
          <span className="decode-stage-arrow" aria-hidden="true" />
          <div className="decode-sequence-stage">
            <div className="decode-real-token-lane">
              <label>Effective Sequence</label>
              <div>
                {compressedPrefillTokens.map((token) => (
                  <i className={token.merged ? "decode-real-super-token" : ""} key={`normal-word-${token.index}`}>
                    <span>{token.word}</span>
                  </i>
                ))}
                <i className="decode-real-new-token"><span>is</span></i>
              </div>
            </div>
            <div className="decode-real-kv-lane">
              <label>KV Cache</label>
              <div>
                {compressedPrefillTokens.map((token) => (
                  <i className={token.merged ? "decode-real-super-kv" : ""} key={`normal-real-kv-${token.index}`}><b>K</b><b>V</b></i>
                ))}
                <i className="decode-empty-kv"><span>No KV</span></i>
              </div>
            </div>
          </div>
        </div>

        <div className="decode-normal-tail-row">
          <div className="decode-side-label decode-simple-label"><strong>Tail Check</strong></div>
          <div className="decode-tail-scan-wide">
            <div className="decode-tail-check-heading">Check token-ID suffixes against Merge Rules R</div>
            <div className="decode-tail-words">
              <i>in</i><i>the</i><i>world</i><i>is</i>
            </div>
            <span className="decode-tail-flow-arrow" aria-hidden="true" />
            <div className="decode-tail-probes">
              <span><b>2-gram</b><i>world · is</i></span>
              <span><b>3-gram</b><i>the · world · is</i></span>
              <span><b>4-gram</b><i>in · the · world · is</i></span>
            </div>
            <span className="decode-tail-flow-arrow" aria-hidden="true" />
            <strong>No suffix appears in R</strong>
          </div>
        </div>

        <div className="decode-forward-block decode-normal-forward">
          <span className="decode-normal-route" aria-hidden="true"><i /></span>
          <div className="decode-side-label decode-simple-label"><strong>Base LLM Forward</strong></div>
          <div className="decode-normal-forward-flow">
            <div className="decode-embedding-card decode-word-embedding"><small>Embedding</small><strong>“is”</strong><em>e<sub>new</sub></em></div>
            <span className="decode-input-plus">+</span>
            <div className="decode-context-input">
              <strong>KV Cache Context</strong>
              <div className="decode-context-memory" aria-label="Cached prefix context">
                <i className="decode-memory-plate decode-memory-back" aria-hidden="true" />
                <i className="decode-memory-plate decode-memory-middle" aria-hidden="true" />
                <span className="decode-memory-ribbon" aria-hidden="true" />
              </div>
            </div>
            <span className="decode-flow-arrow"><i /></span>
            <div className="decode-network-field"><TeacherNeuralField active={active && step === 1} flow="collective" /></div>
          </div>
        </div>

        <div className="decode-normal-output-row">
          <div className="decode-side-label decode-simple-label"><strong>Decoding Output</strong></div>
          <div className="decode-output-pair decode-normal-output-pair">
            <div className="decode-new-kv">
              <span><i>K</i><i>V</i></span>
              <strong>KV for “is”</strong>
            </div>
            <div className="decode-next-logits decode-explicit-prediction">
              <strong>Next-token Prediction</strong>
              <div className="decode-next-prediction-body">
                <span>{decodeDistribution.slice(0, 5).map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</span>
                <i aria-hidden="true">→</i>
                <b>“a”</b>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="decode-frame decode-merge-frame decode-merge-aligned-frame" aria-label="Autoregressive decoding with tail matching, merge, and KV cache rollback">
        <div className="decode-frame-heading">
          <div><span>Decoding</span></div>
          <div className="decode-counts"><span>Token IDs <b>8 → 9</b></span><span>KV length <b>8 → 8</b></span></div>
        </div>

        <div className="decode-normal-state decode-merge-state">
          <div className="decode-side-label decode-simple-label"><strong>Sample &amp; Append</strong></div>
          <div className="decode-prefill-sample">
            <strong>Next-token Prediction</strong>
            <div>{decodeDistribution.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</div>
            <span>sample</span>
            <b>“a”</b>
          </div>
          <span className="decode-merge-stage-arrow" aria-hidden="true" />
          <div className="decode-sequence-stage">
            <div className="decode-real-token-lane">
              <label>Effective Sequence</label>
              <div>
                {decodedTokens.map((token) => (
                  <i className={token.merged ? "decode-real-super-token" : ""} key={`merge-word-${token.index}`}>
                    <span>{token.word}</span>
                  </i>
                ))}
                <i className="decode-real-new-token"><span>a</span></i>
              </div>
            </div>
            <div className="decode-real-kv-lane">
              <label>KV Cache</label>
              <div>
                {decodedTokens.map((token) => (
                  <i className={token.merged ? "decode-real-super-kv" : ""} key={`merge-real-kv-${token.index}`}><b>K</b><b>V</b></i>
                ))}
                <i className="decode-empty-kv"><span>No KV</span></i>
              </div>
            </div>
          </div>
        </div>

        <div className="decode-merge-action-row">
          <div className="decode-side-label decode-simple-label"><strong>Tail Match</strong></div>
          <div className="decode-merge-action-stage">
            <div className="decode-tail-hit-panel">
              <strong>Tail Check</strong>
              <div className="decode-merge-tail-words"><i>the</i><i>world</i><i className="matched">is</i><i className="matched">a</i></div>
              <span><b>2-gram in R</b><em>is · a</em></span>
            </div>
            <span className="decode-action-arrow" aria-hidden="true" />
            <div className="decode-rollback-visual">
              <strong>KV Rollback</strong>
              <div><span className="decode-memory-mini" /><b className="decode-kv-retire">KV for “is”</b></div>
              <small>Remove the previous suffix state</small>
            </div>
            <span className="decode-action-arrow" aria-hidden="true" />
            <div className="decode-merge-module-visual">
              <strong>Apply Mφ</strong>
              <div><span>is</span><i className="decode-pair-amp">&amp;</i><span>a</span><i>→</i><b>Mφ</b><i>→</i><em>“is a”</em></div>
            </div>
          </div>
        </div>

        <div className="decode-forward-block decode-normal-forward decode-merge-forward">
          <span className="decode-merge-route" aria-hidden="true"><i /></span>
          <div className="decode-side-label decode-simple-label"><strong>Base LLM Forward</strong></div>
          <div className="decode-normal-forward-flow">
            <div className="decode-embedding-card decode-word-embedding decode-super-embedding"><small>Merged Embedding</small><strong>“is a”</strong><em>ẽ<sub>is,a</sub></em></div>
            <span className="decode-input-plus">+</span>
            <div className="decode-context-input decode-rollback-context">
              <strong>Rolled-back KV Context</strong>
              <div className="decode-context-memory" aria-label="KV context after suffix rollback">
                <i className="decode-memory-plate decode-memory-back" aria-hidden="true" />
                <i className="decode-memory-plate decode-memory-middle" aria-hidden="true" />
                <span className="decode-memory-ribbon" aria-hidden="true" />
              </div>
            </div>
            <span className="decode-flow-arrow"><i /></span>
            <div className="decode-network-field"><TeacherNeuralField active={active && step === 2} flow="collective" /></div>
          </div>
        </div>

        <div className="decode-merge-output-row">
          <div className="decode-side-label decode-simple-label"><strong>Decoding Output</strong></div>
          <div className="decode-output-pair decode-normal-output-pair decode-merge-output-pair">
            <div className="decode-new-kv">
              <span><i>K</i><i>V</i></span>
              <strong>KV for “is a”</strong>
            </div>
            <div className="decode-next-logits decode-explicit-prediction">
              <strong>Next-token Prediction</strong>
              <div className="decode-next-prediction-body">
                <span>{decodeDistribution.slice(0, 5).map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</span>
                <i aria-hidden="true">→</i>
                <b>“test”</b>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="prefill-explanation">
        <div className="demo-step-meter" aria-label="Select and pause an inference frame">
          {inferenceCaptions.map((caption, index) => (
            <button
              key={caption}
              type="button"
              className={index === step ? "active" : index < step ? "complete" : ""}
              aria-label={`Show inference frame ${index + 1}: ${caption}`}
              aria-pressed={paused && index === step}
              title={`Inference frame ${index + 1}`}
              onClick={() => onSelectStep(index)}
            />
          ))}
          {paused && <button type="button" className="demo-step-resume" onClick={onResume}>Play</button>}
        </div>
        <p><span>0{step + 1}</span><strong>{step === 0 ? "Prefill" : "Decoding:"}</strong>{inferenceCaptions[step]}</p>
      </div>
    </div>
  );
}

export function TrainingStory() {
  const miningSectionRef = useRef<HTMLElement>(null);
  const distillationSectionRef = useRef<HTMLElement>(null);
  const inferenceSectionRef = useRef<HTMLElement>(null);
  const [distillStep, setDistillStep] = useState(0);
  const [distillPaused, setDistillPaused] = useState(false);
  const [prefillActive, setPrefillActive] = useState(false);
  const [inferenceStep, setInferenceStep] = useState(0);
  const [inferencePaused, setInferencePaused] = useState(false);
  const [scanStage, setScanStage] = useState(-1);

  useEffect(() => {
    if (distillPaused) return;
    const section = distillationSectionRef.current;
    if (!section) return;

    let timer: number | undefined;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const durations = [3200, 3600, 4600];
    const stop = () => {
      if (timer !== undefined) window.clearTimeout(timer);
      timer = undefined;
    };
    const playStep = (step: number) => {
      setDistillStep(step);
      if (reducedMotion) return;
      timer = window.setTimeout(() => playStep((step + 1) % distillCaptions.length), durations[step]);
    };
    const start = () => {
      stop();
      playStep(0);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) start();
        else stop();
      },
      { threshold: 0.12 }
    );

    observer.observe(section);
    return () => {
      stop();
      observer.disconnect();
    };
  }, [distillPaused]);

  useEffect(() => {
    const section = miningSectionRef.current;
    if (!section) return;

    let timer: number | undefined;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const stop = () => {
      if (timer !== undefined) window.clearInterval(timer);
      timer = undefined;
    };
    const playStage = (stage: number) => {
      setScanStage(stage);
      if (reducedMotion) return;

      const nextStage = stage === 2 ? -1 : stage + 1;
      const delay = stage === 2 ? 1650 : stage === -1 ? 380 : 1350;
      timer = window.setTimeout(() => playStage(nextStage), delay);
    };
    const start = () => {
      stop();
      playStage(0);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) start();
        else stop();
      },
      { threshold: 0.08 }
    );

    observer.observe(section);
    return () => {
      stop();
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const section = inferenceSectionRef.current;
    if (!section) return;

    let timer: number | undefined;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const durations = [8400, 6100, 7200];
    const stop = () => {
      if (timer !== undefined) window.clearTimeout(timer);
      timer = undefined;
    };
    const playStep = (step: number) => {
      setInferenceStep(step);
      if (reducedMotion || inferencePaused) return;
      timer = window.setTimeout(() => playStep((step + 1) % inferenceCaptions.length), durations[step]);
    };
    const start = () => {
      stop();
      if (!inferencePaused) playStep(0);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        setPrefillActive(entry.isIntersecting);
        if (entry.isIntersecting) start();
        else stop();
      },
      { threshold: 0.12 }
    );
    observer.observe(section);
    return () => {
      stop();
      observer.disconnect();
    };
  }, [inferencePaused]);

  return (
    <>
      <section
        className="training-story mining-story-section"
        id="training"
        ref={miningSectionRef}
        aria-labelledby="training-title"
      >
        <div className="training-sticky">
          <div className="training-shell">
            <header className="training-heading">
              <div className="heading-copy mining-heading">
                <p><span>01</span> Build Merge Rules</p>
                <h2 id="training-title">Extract high-frequency N-grams.</h2>
              </div>
            </header>

            <div className="training-stage">
              <div
                className="training-scene mining-scene"
                data-scan-stage={scanStage}
              >
              <div className="corpus-meta">
                <strong><i /> Detecting repeated spans</strong>
                <b>2 · 3 · 4-grams</b>
              </div>

              <div className="corpus-window">
                <div className="corpus-page">
                  <div className="corpus-copy">
                    <p>
                      A <CorpusHit size={2} order={0}>language model</CorpusHit> learns
                      regularities from large collections of text. Modern <CorpusHit size={2} order={1}>language models</CorpusHit> use
                      those regularities to predict what comes next, creating representations
                      that remain useful <CorpusHit size={3} order={2}>in the world</CorpusHit> beyond
                      the training corpus. Researchers study how a <CorpusHit size={2} order={3}>language model</CorpusHit> preserves
                      meaning across long sequences and how repeated structures appear <CorpusHit size={3} order={4}>in the world</CorpusHit> of
                      natural text. Performance improves <CorpusHit size={4} order={5}>as a result of</CorpusHit> better
                      data and optimization; compression becomes possible <CorpusHit size={4} order={6}>as a result of</CorpusHit> patterns
                      that recur again and again. Across many documents, the same <CorpusHit size={2} order={7}>language model</CorpusHit> encounters
                      familiar structures <CorpusHit size={3} order={8}>in the world</CorpusHit> of news, science, and history. These spans
                      become reliable compression candidates <CorpusHit size={4} order={9}>as a result of</CorpusHit> their repeated use.
                    </p>
                    <p>
                      Large text collections contain biographies, news reports, scientific articles,
                      and conversations written in many different styles. A <CorpusHit size={2} order={10}>language model</CorpusHit> may
                      encounter the same short constructions thousands of times even when the surrounding
                      topic changes. The phrase <CorpusHit size={2} order={11}>language model</CorpusHit> often appears in technical writing,
                      while <CorpusHit size={3} order={12}>in the world</CorpusHit> recurs throughout narrative text and
                      <CorpusHit size={4} order={13}>as a result of</CorpusHit> frequently connects an observation to its consequence.
                      Counting these repetitions across the corpus reveals contiguous token spans that
                      remain stable across documents, authors, and domains.
                    </p>
                    <p>
                      Similar repetition appears across topics that otherwise share very little vocabulary.
                      An astronomy article may describe objects <CorpusHit size={3} order={14}>in the world</CorpusHit> beyond Earth,
                      while a report on computation discusses how a <CorpusHit size={2} order={15}>language model</CorpusHit> processes
                      a sequence. Historical writing links causes and consequences <CorpusHit size={4} order={16}>as a result of</CorpusHit> earlier
                      events, and educational material returns to the same explanation in several chapters.
                      When these local patterns recur at scale, their frequency distinguishes stable spans
                      from combinations that appear only once. The corpus therefore provides both the
                      candidate phrases and enough examples to observe them in varied surroundings.
                    </p>
                  </div>
                </div>
              </div>

              <div className="rule-crystals" aria-label="Discovered merge rules">
                <p>
                  <span>Frequently recurring spans are retained as</span>
                  Merge Rules <b>R</b>
                </p>
                <div className="rule-grid">
                  <div data-size="2" className={scanStage >= 0 ? "active" : ""}>
                    <span>2-gram</span><strong>language model</strong><i />
                  </div>
                  <div data-size="3" className={scanStage >= 1 ? "active" : ""}>
                    <span>3-gram</span><strong>in the world</strong><i />
                  </div>
                  <div data-size="4" className={scanStage >= 2 ? "active" : ""}>
                    <span>4-gram</span><strong>as a result of</strong><i />
                  </div>
                </div>
              </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <MergeModuleStory />

      <section
        className="training-story distillation-story-section"
        id="train-merge-module"
        ref={distillationSectionRef}
        aria-labelledby="distillation-title"
      >
        <div className="training-sticky">
          <div className="training-shell">
            <header className="training-heading">
              <div className="heading-copy distill-heading">
                <p><span>03</span> Train Mφ</p>
                <h2 id="distillation-title">Train Merge Module</h2>
              </div>
            </header>

            <div className="training-stage">
              <div className="training-scene distillation-scene">
                <DistillationAnimation
                  step={distillStep}
                  paused={distillPaused}
                  onSelectStep={(step) => {
                    setDistillStep(step);
                    setDistillPaused(true);
                  }}
                  onResume={() => setDistillPaused(false)}
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section
        className="training-story inference-story-section"
        id="inference"
        ref={inferenceSectionRef}
        aria-labelledby="inference-title"
      >
        <div className="training-sticky">
          <div className="training-shell">
            <header className="training-heading">
              <div className="heading-copy inference-heading">
                <p><span>04</span> Inference</p>
                <h2 id="inference-title">Inference with UMIM</h2>
              </div>
            </header>

            <div className="training-stage">
              <div className="training-scene inference-scene">
                <PrefillAnimation
                  key={`inference-${inferenceStep}`}
                  active={prefillActive}
                  step={inferenceStep}
                  paused={inferencePaused}
                  onSelectStep={(step) => {
                    setInferenceStep(step);
                    setInferencePaused(true);
                  }}
                  onResume={() => setInferencePaused(false)}
                />
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
