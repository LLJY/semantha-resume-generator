# Data layout

Put normalized markdown project records in `data/projects/`.

Each record should come from a repo-local `resume-project.md` and preserve frontmatter plus bullet candidates.

Example future flow:

- source: `portfolio/repos/<owner>__<repo>/resume-project.md`
- normalized copy: `resume-generator/data/projects/<project-id>.md`
