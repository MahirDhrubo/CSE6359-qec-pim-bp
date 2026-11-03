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
from QUICK_GRID and must be refreshed after the FULL_GRID run.
