# Slides (OWNER: P3)

`slides.md` is a [Marp](https://marp.app) deck: plain Markdown, so it diffs and merges like code.
Figures are linked from `../results/`, so run `make results` (or `make results GRID=full`) first.

Export (Node required, no project dependency):

```bash
npx @marp-team/marp-cli slides/slides.md --allow-local-files -o slides/slides.pdf
npx @marp-team/marp-cli slides/slides.md --allow-local-files -o slides/slides.pptx
```

Or open `slides.md` in VS Code with the "Marp for VS Code" extension for a live preview.

Speaker notes live in `<!-- ... -->` comments under each slide. Numbers tagged `[QUICK]` come
from QUICK_GRID; if FULL_GRID data is merged, rerun `make results GRID=full` and refresh them.
Each slide's notes start with a time budget `[~N min]`; they add up to ~27 min.

## Day-4 checklist

- [ ] Fill in team names on the title slide
- [ ] `make results` and `python scripts/p3_demo.py --record results/demo_transcript.txt`
      on the presenting laptop; export the PDF
- [ ] Rehearsal 1, timed: ______ min (limit 30)
- [ ] Rehearsal 2, timed: ______ min (limit 30)
- [ ] Record the backup demo: screen-record `python scripts/p3_demo.py --pause 1.5`
      (e.g. OBS, or `asciinema rec slides/demo.cast` if installed) and keep the file with the PDF
- [ ] Fallback order if the live demo fails: recording, then `results/demo_transcript.txt`
