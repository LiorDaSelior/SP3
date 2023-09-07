symnmf: symnmf.o symnmf.h
    gcc -ansi -Wall -Wextra -Werror -pedantic-errors -o symnmf symnmf.o

symnmf.o: symnmf.c
    gcc -ansi -Wall -Wextra -Werror -pedantic-errors -c symnmf.c