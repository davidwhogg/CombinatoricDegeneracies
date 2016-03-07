LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: cd.pdf

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
