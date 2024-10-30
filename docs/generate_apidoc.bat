@echo off
sphinx-apidoc -o source/generated ../pydhn -f -t source/_templates/apidoc --module-first --implicit-namespaces --separate
