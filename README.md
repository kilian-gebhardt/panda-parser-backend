# panda-parser-backend

This project contains the `C++` backend of [panda-parser](https://github.com/kilian-gebhardt/panda-parser). It consists of several parts:

- A Hypergraph class for the representation of grammars and charts of different types.
- An implementation of training and parsing algorithms on top of hypergraphs: the exectation maximization algorithm in the inside/outside variant with support for cyclic hypergraphs, the split/merge algorithm, various methods to manipulate weight assignments on hypergraph edges.
- A bottom-up parser for *linear context-free rewriting systems* (LCFRS) [[Vijay-Shanker, Weir, and Joshi, 1987]](https://doi.org/10.3115/981175.981190) designed to compute the complete (unpruned) chart.
- An implementation of a *restricted* version of sequence terms (see [[Seki and Kato, 2008, p. 211]](https://doi.org/10.1093/ietisy/e91-d.2.209)) and *simple definite clause programs* (sDCP) [[Deransart and Maluszynski, 1985]](https://doi.org/10.1016/0743-1066(85)90015-9) generating sequence terms (s-terms).
- An implementation of hybrid trees [[Gebhardt, Nederhof, and Vogler, 2017]](https://doi.org/10.1162/COLI_a_00291).
- Bottom-up parsers for parsing an s-term with a sDCP and for parsing a hybrid tree with a LCFRS/sDCP hybrid grammar.

## Contributors
The software was developed by [Kilian Gebhardt](wwwtcs.inf.tu-dresden.de/~kilian/) and [Markus Teichmann](https://wwwtcs.inf.tu-dresden.de/~teichm/). It is maintained by Kilian Gebhardt.

## Installation/Usage
The software requires [Eigen](http://eigen.tuxfamily.org) and [Boost](https://www.boost.org/) to be installed. 
It can be build with [cmake](https://cmake.org/). The executable files contain toy examples for testing the algorithms. Most code is in the various header files which are meant to be used by [panda-parser](https://github.com/kilian-gebhardt/panda-parser). 

You may use **panda-parser-backend** independently of panda-parser in your own project in accordance to the LICENSE.