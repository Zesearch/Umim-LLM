const resultRows = [
  {
    backbone: "Llama-3-8B",
    original: ["13.4", "15.3", "9.2"],
    umim: [
      { ppl: "13.8", reduction: "36.1" },
      { ppl: "15.3", reduction: "14.8" },
      { ppl: "9.8", reduction: "13.8" },
    ],
  },
  {
    backbone: "Llama-3.2-1B",
    original: ["20.1", "21.2", "13.4"],
    umim: [
      { ppl: "23.4", reduction: "36.1" },
      { ppl: "22.0", reduction: "14.8" },
      { ppl: "14.7", reduction: "13.8" },
    ],
  },
  {
    backbone: "GPT2-XL",
    original: ["28.6", "27.1", "13.1"],
    umim: [
      { ppl: "37.9", reduction: "35.1" },
      { ppl: "30.0", reduction: "14.1" },
      { ppl: "14.4", reduction: "12.5" },
    ],
  },
];

export function TaskAgnosticResults() {
  return (
    <section className="transfer-results-section" id="task-agnostic-transfer" aria-labelledby="transfer-results-title">
      <div className="transfer-results-shell">
        <header className="transfer-results-heading">
          <p><span>05</span> Results</p>
          <h2 id="transfer-results-title">Task-Agnostic Transfer</h2>
        </header>

        <figure className="academic-results-figure">
          <figcaption className="academic-results-caption">
            <div className="academic-caption-title">
              <span>Base Mφ · Language Modeling</span>
              <strong>Language modeling performance under token reduction.</strong>
            </div>
            <div className="academic-caption-key">
              <p><strong>PPL ↓</strong><span>Perplexity · lower is better</span></p>
              <p><strong>TR ↑</strong><span>Token reduction · higher is better</span></p>
            </div>
          </figcaption>

          <div className="academic-results-scroll">
            <table className="academic-results-table">
              <thead>
                <tr>
                  <th scope="col" rowSpan={2}>Backbone</th>
                  <th scope="col" rowSpan={2}>Method</th>
                  <th scope="colgroup" colSpan={2}><strong>WikiText-103</strong><span>Rule source</span></th>
                  <th scope="colgroup" colSpan={2}><strong>BookCorpus</strong><span>Unseen corpus</span></th>
                  <th scope="colgroup" colSpan={2}><strong>OpenWebText</strong><span>Unseen corpus</span></th>
                </tr>
                <tr>
                  <th scope="col">PPL ↓</th>
                  <th scope="col">TR (%) ↑</th>
                  <th scope="col">PPL ↓</th>
                  <th scope="col">TR (%) ↑</th>
                  <th scope="col">PPL ↓</th>
                  <th scope="col">TR (%) ↑</th>
                </tr>
              </thead>

              {resultRows.map((row) => (
                <tbody key={row.backbone}>
                  <tr>
                    <th scope="rowgroup" rowSpan={2}>{row.backbone}</th>
                    <th scope="row">Original</th>
                    {row.original.map((ppl, index) => (
                      <FragmentCells key={`${row.backbone}-original-${index}`} ppl={ppl} reduction="—" />
                    ))}
                  </tr>
                  <tr className="academic-umim-row">
                    <th scope="row">Merge Module</th>
                    {row.umim.map((result, index) => (
                      <FragmentCells
                        key={`${row.backbone}-umim-${index}`}
                        ppl={result.ppl}
                        reduction={result.reduction}
                        emphasized
                      />
                    ))}
                  </tr>
                </tbody>
              ))}
            </table>
          </div>

          <div className="academic-results-footnote">
            <strong>Evaluation setting.</strong> The same WikiText-derived merge rules and WikiText-trained Base Mφ
            are applied directly to all three corpora. BookCorpus and OpenWebText are unseen transfer corpora;
            the backbone remains frozen and no downstream retraining is used. Original denotes the uncompressed backbone.
          </div>
        </figure>
      </div>
    </section>
  );
}

function FragmentCells({
  ppl,
  reduction,
  emphasized = false,
}: {
  ppl: string;
  reduction: string;
  emphasized?: boolean;
}) {
  return (
    <>
      <td className={emphasized ? "academic-result-value" : undefined}>{ppl}</td>
      <td className={emphasized ? "academic-result-reduction" : "academic-result-dash"}>{reduction}</td>
    </>
  );
}
