"use client";

import { CSSProperties, useEffect, useRef, useState } from "react";
import katex from "katex";

const mergeModuleSteps = [
  "The static embeddings of a matched token span enter the merge module before any Transformer layer.",
  "A learnable query in each head attends over the same processed span and produces one weighted summary.",
  "The head outputs are concatenated and projected back into the Base LLM embedding space.",
  "One surrogate embedding leaves Mφ and replaces the complete span for the frozen Base LLM.",
];

const attentionHeads = [
  { query: "q₁", output: "o₁", language: 0.82, model: 0.52 },
  { query: "q₂", output: "o₂", language: 0.48, model: 0.92 },
  { query: "q₃", output: "o₃", language: 0.72, model: 0.66 },
  { query: "qₕ", output: "oₕ", language: 0.57, model: 0.84 },
];

const mergeModuleFormula = String.raw`
  M_\phi\!\left(e(x_{t:t+n})\right)
  = W^o \left[\mathrm{o}_1;\cdots;\mathrm{o}_h\right]^\top,
  \quad
  \mathrm{o}_i
  = \mathrm{softmax}\!\left(
    \frac{q_i^\top W^{kv} e(x_{t:t+n})}{\sqrt{d_h}}
  \right)
  \left(W^{kv} e(x_{t:t+n})\right)^\top .
`;

const mergeModuleFormulaHtml = katex.renderToString(mergeModuleFormula, {
  displayMode: true,
  output: "mathml",
  throwOnError: false,
  strict: false,
});

function Vector({ className = "" }: { className?: string }) {
  return (
    <span className={`merge-model-vector ${className}`} aria-hidden="true">
      <i /><i /><i /><i /><i /><i /><i /><i />
    </span>
  );
}

export function MergeModuleStory() {
  const sectionRef = useRef<HTMLElement>(null);
  const [step, setStep] = useState(0);
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.22 },
    );
    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible || paused) return;
    const timer = window.setTimeout(() => {
      setStep((current) => (current + 1) % mergeModuleSteps.length);
    }, 4400);
    return () => window.clearTimeout(timer);
  }, [step, paused, visible]);

  return (
    <section
      className={`merge-module-section merge-module-step-${step}`}
      id="merge-module"
      ref={sectionRef}
      aria-labelledby="merge-module-title"
    >
      <div className="merge-module-shell">
        <header className="merge-module-heading">
          <p><span>02</span> Core component</p>
          <h2 id="merge-module-title">The Merge Module Mφ</h2>
        </header>

        <div className="merge-model-stage">
          <div className="merge-model-legend" aria-label="Architecture states">
            <span><i /> Static embeddings</span>
            <span><i /> Trainable Mφ</span>
            <span><i /> Frozen Base LLM</span>
          </div>

          <div className="merge-model-figure">
            <div className="merge-model-input">
              <div className="merge-input-title">
                <span>Matched span</span>
                <strong>Static token embeddings</strong>
              </div>
              <div className="merge-input-token">
                <b>language</b>
                <Vector />
                <span>e<sub>language</sub></span>
              </div>
              <div className="merge-input-token">
                <b>model</b>
                <Vector />
                <span>e<sub>model</sub></span>
              </div>
            </div>

            <span className="merge-model-flow-arrow downward input-arrow" aria-hidden="true" />

            <div className="merge-model-chassis">
              <div className="merge-chassis-heading">
                <div>
                  <span>Merge Module</span>
                  <strong>Mφ</strong>
                </div>
                <p>Single-layer multi-head attention pooling</p>
                <b>Trainable</b>
              </div>

              <div className="merge-module-stack">
                <div className="merge-output-projection">
                  <span>Output projection</span>
                  <strong>W<sup>o</sup></strong>
                  <em>Linear · GELU · LayerNorm</em>
                </div>

                <div className="merge-concat-bus">
                  <span>Concatenate</span>
                  <strong>[ o₁ ; o₂ ; ··· ; oₕ ]</strong>
                </div>

                <div className="merge-attention-chamber">
                  <div className="merge-attention-bank">
                    <div className="merge-bank-title">
                      <span>Attention pooling</span>
                      <em>Learnable queries Q = {`{q₁, q₂, ···, qₕ}`}</em>
                    </div>
                    <div className="merge-head-fan">
                      {attentionHeads.map((head, index) => (
                        <div
                          className="merge-model-head"
                          key={head.query}
                          style={{
                            "--head-index": index,
                            "--language-weight": head.language,
                            "--model-weight": head.model,
                          } as CSSProperties}
                        >
                          <span className="merge-head-query">{head.query}</span>
                          <i className="merge-head-stem" aria-hidden="true" />
                          <div className="merge-head-mixer">
                            <span
                              className="first"
                              style={{ "--token-weight": head.language } as CSSProperties}
                            >L</span>
                            <b>Head {index === attentionHeads.length - 1 ? "h" : index + 1}</b>
                            <span
                              className="second"
                              style={{ "--token-weight": head.model } as CSSProperties}
                            >M</span>
                          </div>
                          <i className="merge-head-stem" aria-hidden="true" />
                          <span className="merge-head-result">{head.output}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="merge-kv-projection">
                  <div>
                    <span>Shared K / V projection</span>
                    <strong>W<sup>kv</sup></strong>
                  </div>
                  <em>Linear · GELU · LayerNorm</em>
                </div>
              </div>
            </div>

            <span className="merge-model-flow-arrow downward output-arrow" aria-hidden="true" />

            <div className="merge-model-output">
              <div className="merge-output-copy">
                <span>One surrogate embedding</span>
                <strong>language model</strong>
              </div>
              <Vector className="merge-output-vector" />
              <span className="merge-model-flow-arrow horizontal" aria-hidden="true" />
              <div className="merge-model-llm">
                <i />
                <span>Frozen Base LLM</span>
                <strong>θ</strong>
              </div>
            </div>
          </div>

          <div
            className="merge-formula-line merge-formula-katex"
            aria-label="Merge module equation from the paper"
            dangerouslySetInnerHTML={{ __html: mergeModuleFormulaHtml }}
          />

          <div className="merge-module-explanation" aria-live="polite">
            <div className="merge-module-meter" aria-label="Select and pause a merge-module frame">
              {mergeModuleSteps.map((caption, index) => (
                <button
                  key={caption}
                  type="button"
                  className={index === step ? "active" : index < step ? "complete" : ""}
                  aria-label={`Show merge-module frame ${index + 1}: ${caption}`}
                  aria-pressed={paused && index === step}
                  onClick={() => {
                    setStep(index);
                    setPaused(true);
                  }}
                />
              ))}
              {paused && <button type="button" className="merge-module-play" onClick={() => setPaused(false)}>Play</button>}
            </div>
            <p><span>0{step + 1}</span>{mergeModuleSteps[step]}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
