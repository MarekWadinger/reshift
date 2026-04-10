TEX      := pdflatex
TEXFLAGS := -interaction=nonstopmode -halt-on-error
BIB      := biber
BIBFLAGS := --quiet

.DEFAULT_GOAL := all

define run-tex
	@$(TEX) $(TEXFLAGS) $(1) > /dev/null 2>&1 \
	  || (echo "Build failed. Log tail:"; tail -30 $(basename $(1)).log; exit 1)
endef

define run-bib
	@$(BIB) $(BIBFLAGS) $(1) 2>&1 | grep -E "^(WARN|ERROR)" || true
endef

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

# ============================================================================
# Setup & Maintenance
# ============================================================================

.PHONY: .river
.river: ## Clone river fork into vendor/ (required for decomposition modules)
	@if [ ! -d vendor/river ]; then \
		mkdir -p vendor && \
		echo "Cloning river fork from MarekWadinger/river..." && \
		git clone https://github.com/MarekWadinger/river.git vendor/river; \
	fi

.PHONY: install
install: .uv .pre-commit .river ## Install the package, dependencies, and pre-commit for local development
	uv sync --frozen --all-extras --all-packages --group dev --group docs
	pre-commit install --install-hooks

.PHONY: sync
sync: .uv ## Update local packages and uv.lock
	uv sync --all-extras --all-packages --group dev --group docs

# ============================================================================
# Python Code Quality
# ============================================================================

.PHONY: format
format: ## Format the code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint the code
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck
typecheck: ## Run static type checking
	uv run ty .

.PHONY: test
test: ## Run tests without coverage (fast, for local dev)
	uv run pytest

.PHONY: testcov
testcov: ## Run tests with coverage and generate an HTML report
	uv run pytest
	@echo "Coverage report generated at reports/coverage/report/index.html"

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run -a

# ============================================================================
# Notebooks
# ============================================================================

.PHONY: execute-notebooks
execute-notebooks: ## Execute all notebooks in examples/
	jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1

.PHONY: render-notebooks
render-notebooks: ## Render all notebooks to markdown
	jupyter nbconvert --to markdown examples/*.ipynb

# ============================================================================
# LaTeX / Publications
# ============================================================================

.PHONY: build-ecc26
build-ecc26: ## Build ECC26 publication PDF
	@cd publications/ECC26 && \
	echo "[1/4] pdflatex (pass 1)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[2/4] biber..." && \
	$(BIB) $(BIBFLAGS) root 2>&1 | grep -E "^(WARN|ERROR)" || true && \
	echo "[3/4] pdflatex (pass 2)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[4/4] pdflatex (pass 3)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "Done: ECC26/root.pdf"

.PHONY: build-eswa
build-eswa: ## Build ESWA publication PDF
	@cd publications/ESWA && \
	echo "[1/4] pdflatex (pass 1)..." && \
	$(TEX) $(TEXFLAGS) submission.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[2/4] biber..." && \
	$(BIB) $(BIBFLAGS) submission 2>&1 | grep -E "^(WARN|ERROR)" || true && \
	echo "[3/4] pdflatex (pass 2)..." && \
	$(TEX) $(TEXFLAGS) submission.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[4/4] pdflatex (pass 3)..." && \
	$(TEX) $(TEXFLAGS) submission.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "Done: ESWA/submission.pdf"

.PHONY: build-ieeecss
build-ieeecss: ## Build IEEECSS publication PDF
	@cd publications/IEEECSS && \
	echo "[1/4] pdflatex (pass 1)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[2/4] biber..." && \
	$(BIB) $(BIBFLAGS) root 2>&1 | grep -E "^(WARN|ERROR)" || true && \
	echo "[3/4] pdflatex (pass 2)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "[4/4] pdflatex (pass 3)..." && \
	$(TEX) $(TEXFLAGS) root.tex > /dev/null 2>&1 || (echo "Build failed"; exit 1) && \
	echo "Done: IEEECSS/root.pdf"

.PHONY: build-tex
build-tex: build-ecc26 build-eswa build-ieeecss ## Build all publication PDFs

## Single pdflatex pass (no bibliography)
.PHONY: quick
quick:
	@echo "pdflatex..."
	$(call run-tex,$(MAIN).tex)
	@echo "Done: $(MAIN).pdf"

## Rebuild bibliography only
.PHONY: bib
bib:
	$(call run-bib,$(MAIN))

## Word count (body text only, excludes preamble/bibliography)
.PHONY: count
count:
	$(TEXCOUNT) -inc -total content/*.tex

## Generate a diff PDF against a previous commit
## Usage: make diff REV=abc1234
##        make diff REV=HEAD~3
.PHONY: diff
diff:
ifndef REV
	$(error Usage: make diff REV=<git-ref>)
endif
	@git worktree add /tmp/thesis-diff-old $(REV) --detach --quiet
	@latexdiff --flatten /tmp/thesis-diff-old/$(MAIN).tex $(MAIN).tex > $(MAIN)-diff.tex
	@git worktree remove /tmp/thesis-diff-old --force
	@echo "[1/3] pdflatex diff..."
	$(call run-tex,$(MAIN)-diff.tex)
	@echo "[2/3] biber diff..."
	$(call run-bib,$(MAIN)-diff)
	@echo "[3/3] pdflatex diff (pass 2+3)..."
	$(call run-tex,$(MAIN)-diff.tex)
	$(call run-tex,$(MAIN)-diff.tex)
	@echo "Done: $(MAIN)-diff.pdf"

## Freeze: detect TeX packages used by this thesis → texpackages.txt
## Requires a prior build so that .fls exists
.PHONY: freeze
freeze: $(MAIN).fls
	@grep "^INPUT.*/texmf-dist/" $(MAIN).fls | \
	  sed 's|.*texmf-dist/||' | \
	  awk -F'/' '{ \
	    if ($$1 == "tex")        print $$3; \
	    else if ($$1 == "fonts") print $$4; \
	    else                     print $$2; \
	  }' | grep -v '[.]' | sort -u | while read pkg; do \
	    tlmgr info --only-installed "$$pkg" 2>/dev/null | grep -q "^package:" && echo "$$pkg"; \
	  done > $(TEXREQ)
	@echo "biber" >> $(TEXREQ)
	@sort -u -o $(TEXREQ) $(TEXREQ)
	@echo "Frozen $$(wc -l < $(TEXREQ) | tr -d ' ') packages → $(TEXREQ)"

$(MAIN).fls:
	@$(TEX) -recorder $(TEXFLAGS) $(MAIN).tex > /dev/null 2>&1

## Install TeX packages from texpackages.txt + texpackages-dev.txt
.PHONY: texinstall
texinstall: $(TEXREQ) $(TEXDEV)
	sudo tlmgr install $$(cat $(TEXREQ) $(TEXDEV))

## Check figures: report missing and unused figure files
## Output uses file:line format for terminal clickability
.PHONY: ceck-figures
check-figures:
	@echo "=== Missing figures (referenced but not on disk) ==="
	@grep -rn 'includegraphics' content/ --include='*.tex' | \
	  sed 's/.*includegraphics[^{]*{//' | sed 's/}.*//' > /tmp/_fig_refs.txt; \
	  grep -rn 'includegraphics' content/ --include='*.tex' | \
	  paste -d'|' - /tmp/_fig_refs.txt | while IFS='|' read -r src fig; do \
	    fig=$$(echo "$$fig" | sed 's|^figures/||'); \
	    loc=$$(echo "$$src" | cut -d: -f1-2); \
	    if [ ! -f "figures/$$fig" ]; then \
	      echo "  $$loc  $$fig"; \
	    fi; \
	  done; rm -f /tmp/_fig_refs.txt
	@echo ""
	@echo "=== Unused figures (on disk but not referenced) ==="
	@find figures -maxdepth 1 -type f \( -name '*.pdf' -o -name '*.png' \) | sort | while read -r f; do \
	  b=$$(basename "$$f"); \
	  grep -rq "$$b" content/ --include='*.tex' || echo "  $$f"; \
	done

## Check markdown cross-links: verify all relative links resolve
.PHONY: check-links
check-links:
	@echo "=== Broken links in .md files ==="
	@.hooks/check-md-links.sh $$(find . -name '*.md' -not -path './templates/*')

.PHONY: clean-tex
clean-tex: ## Remove LaTeX build artifacts
	rm -f publications/*/*.aux publications/*/*.bbl publications/*/*.bcf publications/*/*.blg \
	      publications/*/*.fdb_latexmk publications/*/*.fls publications/*/*.lof publications/*/*.log \
	      publications/*/*.lot publications/*/*.toc publications/*/*.run.xml publications/*/*.synctex.gz \
	      publications/*/*.bbl-SAVE-ERROR publications/*/*.ind publications/*/*.idx

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs
docs: ## Build the documentation
	uv run mkdocs build

.PHONY: docs-serve
docs-serve: ## Build and serve the documentation locally
	uv run mkdocs serve

# ============================================================================
# Validation & Cleanup
# ============================================================================

.PHONY: validate
validate: format lint typecheck test ## Run code formatting, linting, type checks, and tests

.PHONY: all
all: format lint typecheck testcov ## Run code formatting, linting, type checks, and tests with coverage

.PHONY: clean
clean: clean-tex ## Remove build artifacts, cache files, and coverage reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf reports/
	rm -rf .coverage htmlcov/
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.lof *.log *.lot \
	      *.toc *.run.xml *.synctex.gz *.bbl-SAVE-ERROR indent.log \
	      $(MAIN)-diff.tex $(MAIN)-diff.pdf
	rm -f content/*.aux

.PHONY: help
help: ## Show this help (usage: make help)
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-25s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
