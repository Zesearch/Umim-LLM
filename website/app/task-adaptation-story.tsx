"use client";

import { useEffect, useState } from "react";

const stages = ["New rules", "Task SFT", "DPO"] as const;

function ProbabilityBars({ variant }: { variant: "teacher" | "student" }) {
  const heights = variant === "teacher" ? [32, 62, 88, 49, 26] : [35, 59, 84, 52, 29];

  return (
    <div className={`adapt-probability-bars adapt-${variant}`} aria-hidden="true">
      {heights.map((height, index) => (
        <i key={index} style={{ height: `${height}%` }} />
      ))}
    </div>
  );
}

export function TaskAdaptationStory() {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActiveStage((stage) => (stage + 1) % stages.length);
    }, 5200);

    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="task-adaptation-section" id="task-adaptation" aria-labelledby="task-adaptation-title">
      <div className="task-adaptation-shell">
        <header className="task-adaptation-heading">
          <div>
            <p><span>06</span> Lightweight Adaptation</p>
            <h2 id="task-adaptation-title">Task Adaptation</h2>
          </div>
        </header>

        <figure className="task-adaptation-figure">
          <div className="adapt-frozen-rail">
            <span className="adapt-lock" aria-hidden="true"><i /></span>
            <strong>Base LLM θ</strong>
            <span>Frozen throughout</span>
            <i className="adapt-rail-line" aria-hidden="true" />
          </div>

          <div className="adapt-story-grid">
            <article className={`adapt-stage adapt-rules-stage ${activeStage === 0 ? "is-active" : ""}`}>
              <header className="adapt-stage-heading">
                <span>01</span>
                <div>
                  <small>Target task</small>
                  <h3>Re-mine Merge Rules</h3>
                </div>
              </header>

              <div className="adapt-corpus" aria-label="Repeated spans found in target-task examples">
                <div className="adapt-paper adapt-paper-back" aria-hidden="true" />
                <div className="adapt-paper adapt-paper-middle" aria-hidden="true" />
                <div className="adapt-paper adapt-paper-front">
                  <span>A woman walks into the room and</span>
                  <span>the <mark>language model</mark> predicts</span>
                  <span>what happens <mark>as a result of</mark> it.</span>
                  <span>The answer follows <mark>in the world</mark>...</span>
                </div>
                <div className="adapt-scan-line" aria-hidden="true" />
              </div>

              <div className="adapt-rule-extraction">
                <span>Fast frequency count</span>
                <i aria-hidden="true" />
                <strong>R<sub>task</sub></strong>
              </div>

              <div className="adapt-coverage">
                <div>
                  <span>Task coverage</span>
                  <strong>Broader</strong>
                </div>
                <div className="adapt-coverage-track" aria-hidden="true"><i /></div>
              </div>
            </article>

            <article className={`adapt-stage adapt-sft-stage ${activeStage === 1 ? "is-active" : ""}`}>
              <header className="adapt-stage-heading">
                <span>02</span>
                <div>
                  <small>Supervised adaptation</small>
                  <h3>Task Distillation</h3>
                </div>
              </header>

              <div className="adapt-sft-init">
                <div className="adapt-module adapt-module-base">
                  <small>WikiText initialization</small>
                  <strong>Base M<sub>φ</sub></strong>
                </div>
                <span aria-hidden="true">→</span>
                <div className="adapt-module adapt-module-sft">
                  <small>Trainable</small>
                  <strong>M<sub>φ</sub><sup>SFT</sup></strong>
                </div>
              </div>

              <div className="adapt-distillation-panel">
                <div className="adapt-distribution-row">
                  <div>
                    <span>Teacher</span>
                    <small>Original task sequence</small>
                  </div>
                  <ProbabilityBars variant="teacher" />
                </div>
                <div className="adapt-alignment" aria-hidden="true">
                  <i /><span>Align distributions</span><i />
                </div>
                <div className="adapt-distribution-row adapt-student-row">
                  <div>
                    <span>Student</span>
                    <small>Compressed with R<sub>task</sub></small>
                  </div>
                  <ProbabilityBars variant="student" />
                </div>
              </div>

              <div className="adapt-gradient-note">
                <span aria-hidden="true">↑</span>
                <strong>Only M<sub>φ</sub> receives gradients</strong>
              </div>
            </article>

            <article className={`adapt-stage adapt-dpo-stage ${activeStage === 2 ? "is-active" : ""}`}>
              <header className="adapt-stage-heading">
                <span>03</span>
                <div>
                  <small>Preference optimization</small>
                  <h3>RL with DPO</h3>
                </div>
              </header>

              <div className="adapt-dpo-models">
                <div className="adapt-policy-model">
                  <small>Policy · Trainable</small>
                  <strong>M<sub>φ</sub><sup>SFT</sup></strong>
                </div>
                <div className="adapt-reference-model">
                  <small>Reference · Frozen</small>
                  <strong>M<sub>φ</sub><sup>ref</sup></strong>
                </div>
              </div>

              <div className="adapt-preference-panel">
                <div className="adapt-context-line">
                  <span>Shared compressed context</span>
                  <i aria-hidden="true" />
                </div>
                <div className="adapt-ending adapt-ending-preferred">
                  <span aria-hidden="true">✓</span>
                  <div><small>Preferred</small><strong>Correct continuation</strong></div>
                  <em>log p ↑</em>
                </div>
                <div className="adapt-ending adapt-ending-rejected">
                  <span aria-hidden="true">×</span>
                  <div><small>Rejected</small><strong>Incorrect continuations</strong></div>
                  <em>log p ↓</em>
                </div>
              </div>

              <div className="adapt-final-module">
                <span>DPO preference</span>
                <i aria-hidden="true">→</i>
                <strong>Task-Adapted M<sub>φ</sub></strong>
              </div>
            </article>
          </div>

          <div className="adapt-outcome-row">
            <div>
              <span>More task-specific spans</span>
              <strong>Higher merge ratio</strong>
            </div>
            <i aria-hidden="true" />
            <div>
              <span>Better task behavior</span>
              <strong>Accuracy ↑</strong>
            </div>
            <p>Backbone parameters updated: <strong>0</strong></p>
          </div>

          <figcaption>
            Re-count task-specific merge rules, initialize from the WikiText-trained Base M<sub>φ</sub>,
            then adapt only the lightweight merge module with task distillation and DPO.
          </figcaption>

          <div className="adapt-stage-controls" aria-label="Task adaptation stages">
            {stages.map((stage, index) => (
              <button
                type="button"
                key={stage}
                className={activeStage === index ? "is-active" : ""}
                onClick={() => setActiveStage(index)}
                aria-pressed={activeStage === index}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                {stage}
              </button>
            ))}
          </div>
        </figure>
      </div>
    </section>
  );
}
