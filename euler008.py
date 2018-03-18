# -*- coding: utf-8 -*-
from eulertools import *

problem = 8

        
def main():
    answer = 0
    n      = 13;
    with open("files/euler8.txt",'r') as FILE:
        string = FILE.read();
        
    #{
    numbers = [map(int, x) for x in string.split('0') if len(x) >= n];
    
    for number in numbers:
        x = product(number[:n]);
        for i in range(len(number)-n):
            x = (x/number[i])*number[i+n];
            if x>answer:
                answer = x;
                continue;
            else:
                continue;
    #}
    return answer

##def substrings(s,n):
##    return [s[i:i+n] for i in range(len(s)-n+1)];
##
##def mian():
##    answer = 0
##    n      = 13;
##    with open("files/euler8.txt",'r') as FILE:
##        string = FILE.read();
##    #{
##    numbers = [map(int, x) for x in string.split("0") if len(x) >= n];
##    numbers = sum([substrings(s,n) for s in numbers],[]);
##    answer  = max(product(x) for x in numbers);
##    #}
##    return answer


#"""
"<><><><><><><><><><><><>"
if __name__ == '__main__':
    run(main, problem);
"<><><><><><><><><><><><>"
#"""


