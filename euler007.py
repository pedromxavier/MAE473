# -*- coding: utf-8 -*-
from eulertools import *

problem = 7

def main():
    """
    Let Pn be the n-th prime number, for n >=1.
    According to Rosser's theorem:
    Pn > n*log(n)
    This result was improved upon to be:
    n*(log(n) + log(log(n))) > Pn > n*(log(n) + log(log(n)) - 1)

    Here we use the Sieve of Atkin, called as sieve(limit)
    """
    answer = 0

    n     = 10001;
    #{
    limit  = int(n*(log(n) + log(log(n))));
    answer = sieve(limit)[n-1]; 
    #}
    return answer

#"""
"<><><><><><><><><><><><>"
if __name__ == '__main__':
    run(main, problem);
"<><><><><><><><><><><><>"
#"""


