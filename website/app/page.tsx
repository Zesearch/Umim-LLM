import { FileText, GitFork } from "lucide-react";
import { EntanglementField } from "./entanglement-field";
import { TaskAgnosticResults } from "./task-agnostic-results";
import { TaskAdaptationStory } from "./task-adaptation-story";
import { TaskAdaptedResults } from "./task-adapted-results";
import { TrainingStory } from "./training-story";

export default function Home() {
  return (
    <>
      <main className="paper-home">
        <EntanglementField />

        <section className="hero" aria-labelledby="paper-title">
        <p className="venue-line">
          <span className="venue-mark" aria-hidden="true" />
          UMIM
        </p>

        <h1 id="paper-title">
          Distilling Sequential Computation
          <span> in Transformer Language Models</span>
        </h1>

        <div className="authors" aria-label="Paper authors">
          <span>Zixuan Lan<sup>1</sup></span>
          <span>Jessica Yang<sup>2</sup></span>
          <span>Yanhong Li<sup>3</sup></span>
          <span>Karen Livescu<sup>2</sup></span>
          <span>Jiawei Zhou<sup>4</sup></span>
        </div>

        <div className="affiliations" aria-label="Author affiliations">
          <span><sup>1</sup>University of Chicago</span>
          <span><sup>2</sup>Toyota Technological Institute at Chicago</span>
          <span><sup>3</sup>Independent Researcher</span>
          <span><sup>4</sup>Stony Brook University</span>
        </div>

        <nav className="paper-links" aria-label="Paper resources">
          <a
            className="resource-link"
            href="/paper.pdf"
            target="_blank"
            rel="noreferrer"
            aria-label="Read the UMIM paper"
          >
            <span className="icon-shell" aria-hidden="true">
              <FileText strokeWidth={1.55} />
            </span>
            <span>Paper</span>
          </a>

          <a
            className="resource-link"
            href="https://github.com/Zesearch/Umim-LLM"
            target="_blank"
            rel="noreferrer"
            aria-label="View UMIM on GitHub"
          >
            <span className="icon-shell" aria-hidden="true">
              <GitFork strokeWidth={1.55} />
            </span>
            <span>GitHub</span>
          </a>
        </nav>
        </section>

        <div className="scroll-cue" aria-hidden="true">
          <span />
        </div>
      </main>

      <section className="abstract-section" id="abstract" aria-labelledby="abstract-title">
        <div className="abstract-shell">
          <header className="abstract-heading">
            <div>
              <p><span aria-hidden="true" /> Paper overview</p>
              <h2 id="abstract-title">Abstract</h2>
            </div>
          </header>

          <article className="abstract-card">
            <p>
              Transformer language models process sequences token by token in an autoregressive manner,
              making growing contexts increasingly expensive. Yet many adjacent token spans are highly
              predictable or frequently occur as stable units, suggesting that their representations may be
              compressible. We introduce a method for distilling sequential computation by replacing spans of
              input tokens with collapsed representations, computed on the fly by a lightweight merge module.
              This module generates <mark>a single surrogate embedding</mark> from a sequence of static token
              embeddings that captures the functional role of the multiple tokens, allowing pretrained models
              to operate on compressed inputs <mark>without architectural changes or re-training</mark>. We
              apply this approach during inference to compress both prompts and intermediate decoding steps,
              using a rollback mechanism to substitute stored multi-token KV cache entries with their
              single-step surrogates. Experiments across diverse models show that the merge module can be used
              to reduce effective sequence length by up to 40% with minimal accuracy degradation across
              language modeling evaluations and downstream tasks, including question answering, summarization,
              commonsense reasoning, and long-form mathematical reasoning. <mark className="abstract-adaptation">
              Additional lightweight adaptation of the merge module</mark> further improves the
              accuracy-compression trade-off in selected settings. These results demonstrate that sequential
              token computation in Transformers can be effectively approximated through condensed surrogate
              representations that preserve the original behavior without model updating.
            </p>
          </article>
        </div>
      </section>

      <TrainingStory />

      <TaskAgnosticResults />

      <TaskAdaptationStory />

      <TaskAdaptedResults />
    </>
  );
}
