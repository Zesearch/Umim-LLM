import { Database, LockKeyhole, RefreshCw } from "lucide-react";

export function TaskAdaptationStory() {
  return (
    <section className="task-adaptation-section" id="task-adaptation" aria-labelledby="task-adaptation-title">
      <div className="task-adaptation-shell">
        <header className="task-adaptation-heading">
          <div>
            <p><span>06</span> Lightweight Adaptation</p>
            <h2 id="task-adaptation-title">Task Adaptation</h2>
          </div>
        </header>

        <figure className="task-adaptation-figure adapt-method-figure">
          <div className="demo-legend adapt-method-legend">
            <span className="demo-rule-state"><i /> Merge Rules <b>R<sub>task</sub> · Re-mined</b></span>
            <span className="demo-frozen-state"><i /> Base LLM θ <b>Frozen throughout</b></span>
            <span className="demo-trainable-state"><i /> Merge Module Mφ <b>Only trainable component</b></span>
          </div>

          <div className="adapt-method-canvas">
            <section className="adapt-method-stage adapt-method-rules" aria-labelledby="adapt-method-rules-title">
              <header className="adapt-method-stage-label">
                <span>01</span>
                <div>
                  <h3 id="adapt-method-rules-title">Re-mine rules</h3>
                </div>
              </header>

              <div className="adapt-method-stage-body adapt-rule-computation">
                <div className="adapt-target-examples">
                  <div className="adapt-method-component-heading">
                    <small>Target task</small>
                    <strong>D<sub>task</sub></strong>
                  </div>
                  <div className="adapt-target-task-visual">
                    <span className="adapt-target-task-icon" aria-hidden="true"><Database /></span>
                    <div>
                      <strong>Downstream task</strong>
                      <span className="adapt-target-task-samples" aria-hidden="true">
                        <i><b /><b /><b /></i>
                        <i><b /><b /><b /></i>
                        <i><b /><b /><b /></i>
                      </span>
                    </div>
                  </div>
                </div>

                <div className="adapt-method-arrow adapt-method-rule-arrow" aria-hidden="true">
                  <span>count 2–4-grams</span><i />
                </div>

                <div className="adapt-method-miner">
                  <div>
                    <small>Frequency mining</small>
                    <strong>f(g) ≥ τ</strong>
                  </div>
                  <div className="adapt-method-frequency" aria-hidden="true">
                    <span>τ</span>
                    <i style={{ height: "29%" }} />
                    <i style={{ height: "45%" }} />
                    <i className="is-kept" style={{ height: "72%" }} />
                    <i className="is-kept" style={{ height: "88%" }} />
                    <i className="is-kept" style={{ height: "64%" }} />
                    <i style={{ height: "36%" }} />
                  </div>
                </div>

                <div className="adapt-method-arrow" aria-hidden="true"><i /></div>

                <div className="adapt-method-ruleset">
                  <div className="adapt-method-component-heading">
                    <small>Task-specific merge rules</small>
                  </div>
                  <div className="adapt-rule-set-visual">
                    <div className="adapt-rule-patterns" aria-hidden="true">
                      <span><i /><i /><b /><em /></span>
                      <span><i /><i /><i /><b /><em /></span>
                      <span><i /><i /><i /><i /><b /><em /></span>
                    </div>
                    <strong>R<sub>task</sub></strong>
                  </div>
                </div>
              </div>
            </section>

            <div className="adapt-rules-to-sft" aria-hidden="true">
              <i />
            </div>

            <section className="adapt-method-stage adapt-method-sft" aria-labelledby="adapt-method-sft-title">
              <header className="adapt-method-stage-label">
                <span>02</span>
                <div>
                  <h3 id="adapt-method-sft-title">Task SFT</h3>
                </div>
              </header>

              <div className="adapt-method-stage-body adapt-sft-summary">
                <div className="adapt-sft-summary-inputs">
                  <div className="adapt-sft-summary-source">
                    <small>WikiText-pretrained</small>
                    <strong>M<sub>φ</sub><sup>WT</sup></strong>
                    <span>Base merge module</span>
                  </div>
                  <i aria-hidden="true">+</i>
                  <div className="adapt-sft-summary-source is-rules">
                    <small>Task-specific rules</small>
                    <strong>R<sub>task</sub></strong>
                    <span>Re-mined above</span>
                  </div>
                </div>

                <div className="adapt-method-arrow" aria-hidden="true"><i /></div>

                <div className="adapt-sft-summary-process">
                  <small>Target-task adaptation</small>
                  <strong>Task SFT</strong>
                  <div>
                    <span className="is-frozen"><LockKeyhole /> Base LLM θ frozen</span>
                    <span className="is-trainable"><RefreshCw /> Update M<sub>φ</sub> only</span>
                  </div>
                </div>

                <div className="adapt-method-update-arrow">
                  <span>update φ only</span><i aria-hidden="true" />
                </div>

                <div className="adapt-method-module-state is-sft">
                  <small>SFT merge module</small>
                  <strong>M<sub>φ</sub><sup>SFT</sup></strong>
                </div>
              </div>

            </section>

            <div className="adapt-sft-to-rl" aria-hidden="true"><i /></div>

            <section className="adapt-method-stage adapt-method-rl" aria-labelledby="adapt-method-rl-title">
              <header className="adapt-method-stage-label">
                <span>03</span>
                <div>
                  <h3 id="adapt-method-rl-title">Task RL</h3>
                </div>
              </header>

              <div className="adapt-method-stage-body adapt-rl-summary">
                <div className="adapt-method-module-state is-sft adapt-rl-start">
                  <small>SFT merge module</small>
                  <strong>M<sub>φ</sub><sup>SFT</sup></strong>
                  <em>with R<sub>task</sub></em>
                </div>

                <div className="adapt-method-arrow"><span>construct pairs</span><i aria-hidden="true" /></div>

                <div className="adapt-dpo-sequences">
                  <div className="adapt-method-component-heading">
                    <small>DPO sequence pairs</small>
                  </div>
                  <div className="adapt-dpo-pairs" aria-label="Chosen and rejected DPO sequences">
                    <span className="is-chosen"><b>y<sup>+</sup></b><i /><i /><i /></span>
                    <span className="is-rejected"><b>y<sup>−</sup></b><i /><i /><i /></span>
                  </div>
                </div>

                <div className="adapt-method-arrow" aria-hidden="true"><i /></div>

                <div className="adapt-sft-summary-process adapt-rl-summary-process">
                  <small>Task-specific optimization</small>
                  <strong>Task RL (DPO)</strong>
                  <div>
                    <span className="is-frozen"><LockKeyhole /> Base LLM θ frozen</span>
                    <span className="is-trainable"><RefreshCw /> Update M<sub>φ</sub> only</span>
                  </div>
                </div>

                <div className="adapt-method-update-arrow adapt-rl-update-arrow">
                  <span>update φ only</span><i aria-hidden="true" />
                </div>

                <div className="adapt-method-module-state is-final">
                  <small>Final module</small>
                  <strong>M<sub>φ</sub><sup>task</sup></strong>
                </div>
              </div>
            </section>

            <div className="adapt-method-deployment">
              <span>Final task-adapted inference</span>
              <div className="adapt-deployment-flow">
                <strong>Target-task input</strong>
                <i aria-hidden="true" />
                <strong className="is-merge">Merge token spans using R<sub>task</sub> and M<sub>φ</sub><sup>task</sup></strong>
                <i aria-hidden="true" />
                <strong className="is-frozen">Frozen LLM θ</strong>
              </div>
            </div>
          </div>

          <figcaption>
            Task adaptation re-mines R<sub>task</sub>, applies SFT to the WikiText-pretrained merge module, uses M<sub>φ</sub><sup>SFT</sup> to construct DPO sequence pairs, and applies RL only to the merge module. The backbone LLM remains frozen throughout.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
