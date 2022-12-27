# BLU

BLU is a port of [BASICLU](https://github.com/ERGO-Code/basiclu) to
[Rust](https://www.rust-lang.org/).

## Algorithm

BLU implements a right-looking LU factorization with dynamic Markowitz
search and columnwise threshold pivoting. After a column modification to the
matrix it applies either a permutation or the Forrest-Tomlin update to maintain
a factorized form. It uses the method of Gilbert and Peierls to solve triangular
systems with a sparse right-hand side. A more detailed explanation of the method
is given in [Technical Report ERGO 17-002,
http://www.maths.ed.ac.uk/ERGO/preprints.html].


## License

The BLU source code is distributed under the MIT license ([LICENSE](LICENSE) or
https://opensource.org/licenses/MIT).