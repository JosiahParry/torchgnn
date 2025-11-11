doc:
  R -q -e  'devtools::document()'

check: 
  R -q -e 'devtools::check()'

inst:
  R -q -e 'devtools::document();devtools::install()'

readme:
  quarto render README.qmd