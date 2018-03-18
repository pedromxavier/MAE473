# -*- coding: utf-8 -*-
from eulertools import *

problem = 2

def main():
    """
    Let F(n) be the n-th Fibonacci number.
    Let S(n) be the sum of the n-th first Fibonacci numbers.

    So,
    
    S(n) = F(n+2) - 1;

    and looking at the sequence:
    1,1,2,3,5,8,13,21,34,55,...

    we can see that the sum of the even numbers is the same as
    the odd ones and the sum of even numbers will be S(n)/2 if
    F(n) is even. 

    Also:

    F(n) = round((phi**n)/sqrt(5));

    so:

    n(F) = round(log(F*sqrt(5), phi));

    Since n is our upper bound:

    n(F) = floor(round(log(F*sqrt(5), phi)));
    """
    answer = 0

    bound  = int(4e+6);
    #{
    n = int(floor(log(bound*sqrt(5),phi))) + 1;
    F = lambda n: int(floor((phi**n)/sqrt(5)));
    
    if F(n)%2: 
        if F(n-1)%2:
            answer = (F(n)-1)/2;
        else:
            answer = (F(n+1)-1)/2;
    else:
        answer = (F(n+2)-1)/2;
    
    #}
    return answer

#"""
"<><><><><><><><><><><><>"
if __name__ == '__main__':
    run(main, problem);
"<><><><><><><><><><><><>"
#"""
