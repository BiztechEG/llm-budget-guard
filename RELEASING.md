# Releasing

Releases go to PyPI automatically when a version tag is pushed. `.github/workflows/release.yml` builds the
package, checks it, and publishes it with PyPI trusted publishing, so no API token is stored anywhere.

## One-time setup

1. **PyPI publisher.** Sign in at <https://pypi.org>, open *Your account → Publishing*, and add a
   *pending publisher* (the project doesn't exist on PyPI until the first upload):

   | Field | Value |
   |---|---|
   | PyPI project name | `llm-spend-guard` (must match `name` in `pyproject.toml`) |
   | Owner | `BiztechEG` |
   | Repository name | `llm-budget-guard` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. **GitHub environment (optional but recommended).** In the repo, open *Settings → Environments*, create
   `pypi`, and add yourself under *Required reviewers*. Each release then waits for your click before
   anything is uploaded. Without this step GitHub creates the environment on the first run, with no approval.

## Each release

1. On a branch, set the new version in `src/budget_guard/__init__.py` (`__version__ = "0.1.1"`).
2. In `CHANGELOG.md`, replace `(unreleased)` with today's date and list what changed.
3. Open a PR, wait for CI to pass, and merge it.
4. Tag the merge commit on `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag v0.1.1
   git push origin v0.1.1
   ```

5. Watch the *Release* run under the repo's *Actions* tab (and approve it if you set up required reviewers).
6. Check it installs: `pip install llm-spend-guard==0.1.1`.

The workflow refuses a tag that doesn't match `__version__`, so a typo stops the release instead of
publishing the wrong version.

## Versions

- Patch (`0.1.1`): bug fixes and price table updates.
- Minor (`0.2.0`): new features, such as a new provider or storage backend.
- Before `1.0.0`, a minor release may change the API; say so in the changelog.

PyPI never accepts the same version twice, even after a delete. If a release goes wrong, fix it and
release the next patch version.

## Updating prices

Edit `src/budget_guard/pricing.json` (USD per 1M tokens, values as strings), update `_meta.updated`, run
the tests, and ship it as a patch release.
