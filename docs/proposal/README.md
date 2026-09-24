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

## Before submitting, fill in

- Group number, on the title line
- Vaibhav and Siddh's surnames
- Supervisor name
- The project name if `fpgAI` is not what the team agrees on. Candidates raised so
  far: fpgAI, fpgLLM, LLMArray, distLLM.

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

`proposal.tex` uses the `pgfgantt` package, which Overleaf provides. The chart is a
`figure*`, so it spans both columns and floats to the top of a page. Four module
lanes run concurrently across it, which is the visual evidence for the "modules can
be completed in parallel" criterion, so keep it full width rather than shrinking it
into one column.

It replaced the milestone table. Do not re-add that table; the chart carries the
same information and the page budget will not take both.

**If pgfgantt misbehaves on Overleaf**, the quickest fixes in order:

1. Shrink it: reduce `x unit` to `0.48cm` and `y unit chart` to `0.36cm`
2. If it still overflows the page, drop the `Shared` group and its two bars
3. As a last resort, replace the whole `ganttchart` environment with a plain
   `tabular` of week ranges per module. The parallelism is less obvious but the
   schedule information survives

Estimated length with the chart is about 3.7 pages, so there is a little headroom
under the 4-page limit, but check it first.
