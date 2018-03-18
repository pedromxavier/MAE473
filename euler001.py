# -*- coding: utf-8 -*-
from eulertools import *

problem = 1

def ap(r,a):
    return (r*(r+1)*a)/2;
    
def main():
    answer = 0
    a,b,n  = 3,5,1000;
    #{
    m      = LCM(a,b);
    answer = ap((n-1)//a, a) + ap((n-1)//b,b) - ap((n-1)//m,m);
    #}
    return answer

#"""
"<><><><><><><><><><><><>"
if __name__ == '__main__':
    run(main, problem);
"<><><><><><><><><><><><>"
#"""


