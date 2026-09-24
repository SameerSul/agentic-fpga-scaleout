# Capstone proposal

`proposal.tex` is IEEEtran two-column, matching the format of the example proposal.

## Building it

No LaTeX is installed on the Mac, so use Overleaf:

1. New Project, Upload Project, drop in `proposal.tex`
2. Overleaf pulls `IEEEtran.cls` automatically
3. Compile, then check the page count

**The limit is 4 pages** excluding the cover page, references and appendix, and
"format, length is correct" is a graded criterion. Current draft is about 2,440
words of body text plus four tables, which should land just under. If it runs over,
cut in this order: the Background paragraph on AlphaChip and ChipNeMo, then the
Resources software list, then shorten the module descriptions.

## Title block

Filled in: Group 9, Sameer Suleman, Yax Patel, Vaibhav Gopalakrishnan, Siddh Patel,
Supervisor Dr. Shirani. Project name settled as `fpgAI`.

## Rubric coverage

Each of the ten graded criteria maps to a section:

| Criterion | Section |
|---|---|
| Summary is well written | 1 Executive Summary |
| Need and background, context, stakeholders, similar products | 2 Expression of Need, 3 Background |
| Objectives clearly defined and quantifiable | 4 Objectives, table of 12 measurable targets |
| Technical modules, one per member, completable in parallel | 5 Technical Modules, with the interface contracts that make them separable |
| Progress monitoring and control well defined | 9 Progress Monitoring and Control |
| Format, length correct | IEEEtran, 4 pages, verify on Overleaf |
| Deliverables in line with objectives and modules | 6 Deliverables, Bronze, Silver, Gold |
| Health, sustainability, ethical, security, economic, aesthetics, human factors, regulatory | 10 Broader Considerations, all eight addressed |
| Resources, components, people, budget | 7 Resources, budget table totalling 770 CAD |
| Milestones well defined and achievable | 8 Schedule, twelve two-week milestones |

Due September 27. Presentation slot September 30.

## The Gantt chart

`proposal.tex` uses `pgfgantt`, which Overleaf provides. The chart is a `figure*`,
so it spans both columns and floats to the top of a page. Keep it full width: the
four concurrent module lanes are the visual evidence for the "modules can be
completed in parallel" criterion, and shrinking it into one column destroys that.

It covers the **full 28 weeks, September to April**, with a month header above the
week scale. Tasks overlap both across lanes and within each lane, so no member is
ever shown waiting on a predecessor. Integration testing and bi-weekly reviews run
continuously from week 6, and five weeks of float sit at the end.

It replaced the milestone table. Do not re-add that table; the chart carries the
same information and the page budget will not take both.

Approximate geometry: 34 rows at 12.0 cm tall and 17.5 cm wide, against an IEEE
text width of 18.4 cm. It fits, but not with much to spare.

**If it overflows**, in order:

1. Reduce `y unit chart` from `0.325cm` to `0.30cm`
2. Reduce `x unit` from `0.455cm` to `0.42cm`
3. Drop the `Continuous and shared` group and its four bars
4. Drop one bar from each module lane, taking the last one in each
5. Last resort, replace the `ganttchart` environment with a plain `tabular` of
   week ranges per module. Schedule survives, parallelism becomes less obvious

**Milestone weeks:** Proposal 2, Bronze 8, Silver 19, Gold 26. If the term calendar
shifts these, every bar position is a plain integer and easy to move.
