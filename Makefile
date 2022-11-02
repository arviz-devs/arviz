# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -b html
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ArviZ
SOURCEDIR     = doc/source
BUILDDIR      = doc/build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html:
	sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" -b html

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" -b html

cleandocs:
	rm -r "$(BUILDDIR)" "doc/jupyter_execute" "$(SOURCEDIR)/api/generated" "$(SOURCEDIR)/examples"

preview:
	python -m webbrowser "$(BUILDDIR)/index.html"
