#coding: utf-8

from math       import *;
from time       import clock;
from itertools  import *;
from string     import ascii_lowercase as letters;
from string     import ascii_uppercase as capitals;

version = 1.8

print "import :: EulerTools v%.1f ::" % version

def do(n):
    return repeat(False,n);

def PoissonDistribution(l, x):
    "Poisson's Distribution.\nPD(l, x) --> x in [0,1]\nl = event occurence rate\nx = expected events to occour"
    return (l**x * e**-l)/Factorial(x)
    
def BinomialDistribution(p,n,k):
    "Bernoulli's Binomial Distribution.\nBD(p,n,k) --> x in [0, 1]\np = success probability\nn = attempts\nk = expected successfull attempts."
    return Binome(n,k) * p**(k) * (1-p)**(n-k)
    
def Binome(n, k):
    "Newton's Binome Coeficient."
    return Factorial(n)/(Factorial(n-k)*Factorial(k))

def rFlatten(iterable):
    "flatten([1,[2,3,[4,5,6]]]) --> list([1,2,3,4,5,6])"
    iterable = list(iterable); pool = [];
    for i in iterable:
        if not isinstance(i, list):
            pool.append(i)
        else:
            pool += rFlatten(i)
    return pool


def Chebyshev(n):
    assert n >= 0 and type(n) is int;
    
    if not n:
        return lambda x: 1;
    elif n == 1:
        return lambda x: x;
    else:
        return lambda x: 2*x*Chebyshev(n-1)(x) - Chebyshev(n-2)(x);
        

def is_congruent(a,b, mod):
    return not mod%(a-b)

def concatenate(*ints):
    return int("".join(str(i) for i in ints))

def xsum(*args):
    return sum(args);

def flog(x,b):
    "Floor(Log(x)) --> y ~ number of digits of x"
    return int(log(x,b))

def Divisors(x):
    "Divisors(x) --> set([1, D1, D2, D3,...x])\nDi|x"
    pool = set()
    for i in xrange(1, int(sqrt(x))+1):
        if not x%i:
            pool.add(i)
            pool.add(x/i)
    return pool

def nDivisiors(x):
    return len(Divisors(x))
        
def digit_sum(x, b=10):
    n = 0; dsum = 0;
    y = flog(x,b)
    while n <= y:
        dsum += (x%b**(n+1) - x%b**(n))/b**n;
        n += 1;
    return dsum;

def Sigma(i,n,a=lambda j:j):
    return sum(a(k) for k in xrange(i, n+1))

def Pi(i,n,a=lambda j:j):
    return product(a(k) for k in xrange(i, n+1))
        
def rReverse(x,y=0):
    "Reverse(1234) --> 4321"
    if x < 10:
        return x + y
    else:
        return rReverse(x/10, y*10 + x%10 * 10)

def Reverse(x,b):
    "Reverse(1234) --> 4321"
    y =  0;
    z =  0;
    while x:
        x,z = divmod(x,10)
        y   = y*10 + z
    return y

def Discrep(ref, x):
    return abs(float(ref - x)/float(x))

def Average(*args):
    return float(sum(args))/float(len(args));

def gcd(*args):
    return reduce(GCD, args)

def lcm(*args):
    return reduce(LCM, args);

def GCD(a,b):
    "The Greatest Common Divisor"
    while b:
        a,b = b,a%b
    return a

def bGCD(a,b):
    "The Binary Greatest Common Divisor"
    if a==b:
        return a;
    if not a:
        return b;
    if not b:
        return a;

    if ~a & b:
        if b & 1:
            return bGCD(a >> 1, b);
        else:
            return bGCD(a >> 1, b >> 1) << 1;

    if ~b & 1:
        return bGCD(a, b >> 1);
    if a > b:
        return bGCD((a-b) >> 1, b);

    return bGCD((b - a) >> 1, a);
    
def rGCD(a,b):
    "The Recursive Greatest Common Divisor"
    if not b:
        return a
    else:
        return rGCD(b, a%b)
    
def eGCD(a,b):
    "The Extended Greatest Common Divisor"
    x = [1,0]; y= [0,1];
    while b:
        q, a, b = a//b, b, a%b
        x       = [x[1], x[0] - q*x[1]]
        y       = [y[1], y[0] - q*y[1]]
    return a, x[0], y[0]

def reGCD(a,b):
    "The Recursive Extended Greatest Common Divisor"
    if not a:
        return b, 0 ,1
    else:
        g, x, y = reGCD(b%a,a)
        return g, y - (b/a)*x, x

def LCM(a,b):
    "The Least Common Multiple"
    return (abs(a)//GCD(a,b))*abs(b);

def SolveLinearDiofantine(A,B,C):
    "Solves linear Ax + By = C diofantine equation\n(A, B, C, x & y are integers)\nx0=alpha\ny0=beta"
    d = GCD(A,B)
    e = C / d
    if not C % d:
        _, alpha, beta = eGCD(A, B);
        (x0,y0) = (alpha*e, beta*e)

        def solution(t=0):
            x = x0 + B*t/d
            y = y0 - A*t/d
            return (x,y)

        return solution
        
    else:
        raise ArithmeticError("There's no solution for %dx + %dy = %d" % (A,B,C))

def simplify(a,b):
    d = GCD(a,b)
    while (d-1):
        a/=d
        b/=d
        d = GCD(a,b)
    return (a,b)

def NonNegativeSolution(A,B,C):
    solution = SolveLinearDiofantine(A,B,C)
    t=0; z=1; pool=set()
    while True:
        x,y = solution(t)
        if x>0 and y>0:
            pool.add(t)
        elif x>0 and y<0:
            t+=z
            pass
    
def is_pandigital(x,n=9):
    "Verifies if a number is one-to-n pandigital"
    ram = {'0':1}; s = str(x);
    for digit in s:
        try:
            return not ram[digit]
        except:
            ram[digit] = 1
    return not (len(ram) - (n+1))

def is_palindrome(x):
    s = str(x)
    return s == s[::-1]

def is_prime(x):
    if not(x-2):
        return True
    if x<2 or not (x%2):
        return False   
    for i in xrange(3,int(x**0.5)+1,2):
        if not x%i:
            return False
    return True

def is_coprimes(a,b):
    return not GCD(a,b) - 1

def all_coprimes(*args):
    for a,b in combinations(args, 2):
        if GCD(a,b)-1:
            return False
        else:
            continue
    return True

def Totient(n):
    return product(map(lambda x: (x[0]**x[1]) - (x[0]**(x[1]-1)), pwrFactorate(n)))

def FourierTotient(n):
    "Fourier's Totient function."
    return int(round(sum(GCD(k,n)*cos(2*pi*k/n) for k in xrange(1,n+1))))

def EulerTotient(n):
    "Euler's Totient function."
    return int(round(n*product(((1.0-(1.0/p)) for p in set(Factorate(n, True))))))

def _digit_pattern_(x):
    pattern = {}
    for char in str(x):
        try:
            pattern[char] += 1
        except:
            pattern[char] = 1
    return pattern
        

def same_digits(pattern={},*ints):
    "Tells if two integers contain the same digits."
    return xall(lambda x: x==pattern,*(_digit_pattern_(x) for x in ints))
    

def xprimes(start=0,stop=float('inf')):
    """generator for yielding primes within [start, stop]
    including both ends.
    """
    n = start;pool = [];j=0;i=0;
    if n <=2:
        n=2;
        yield n
        pool=[n]
        j=1;n=3;
    else:
        for x in xprimes(2,n):
            pool.append(x)
            j+=1
        if not n%2:
            n+=1
    while n <= stop:
        while (j>i) and (pool[i] < int(n**0.5)+1):
            if not n%pool[i]:
                break
            else:
                i += 1;
                continue
            break
        else:                
            pool.append(n); j+=1;
            yield n
        n += 2; i = 0;

def xpandigitals(n):
    """xpandigitals(n) --> <generator of one-to-n pandigitals>"""
    for i in permutations(range(1,n+1),n):
        yield sum(i[j]*10**(n-j-1) for j in xrange(n))
        
def rFactorial(n):
    "rFactorial(n) --> n! (recursive)"
    if n in (0,1):
        return 1
    else:
        return n*rFactorial(n-1)

def Factorial(n):
    "rFactorial(n) --> n!"
    i=j=1;
    while n>=i:
        j*=i;i+=1;
    return j

def nFactorate(x,include=False):
    "nFactorate(x) --> number of x's distinct prime factors"
    if not x:return 0
    y=x;j=0;k=0;
    for i in xprimes():
        if y==i:break;
        if x==1:break;
        while not (x%i):
            k=1; x/=i;
        j+=k;k=0;
    return j + include

def Factorate(x,include=False):
    "Factorate(x) --> tuple containing x's prime factors"
    if not x: return ()
    y=x;pool=();k=0;
    sieve_y = sieve(y+1)
    for i in sieve_y:
        if y==i:break;
        if x==1:break;
        while not (x%i):
            x/=i;
            pool += (i,)
    return pool + (y,)*(y in sieve_y)*include

def pwrFactorate(x,include=False):
    "pwrFactorate(x) --> pairs containing x's prime factors\nalong with each respective power"
    pool = []; j = 0;
    for i in Factorate(x,include):
        if (j-i):
            pool.append([i,1])
            j = i
        else:
            pool[-1][1] += 1
    return pool            
        
def Rotations(x):
    "Rotations(abc) --> (abc, bca, cab,)"
    r = int(floor(log10(x)))
    pool = (x,)
    for i in xrange(r):
        x = x/10 + (x%10 * 10**r)
        pool += (x,)
    return pool

def xRotate(x):
    "Yields x, and then its rotations"
    r = int(floor(log10(x)))
    yield x
    for i in xrange(r):
        x = x/10 + (x%10 * 10**r)
        yield x

def xall(func, *args):
    "Applies func to all args and check if every\noutput is True."
    return all(func(x) for x in args);

def sieve(LIMIT):
    "Atkins sieve implementation.\nreturns a list with every prime bellow the limit"
    TEST_LIMIT = int(ceil(sqrt(LIMIT)))
    PRIMES     = []
    SIEVE      = [False]*(LIMIT+1)

    for i in xrange(TEST_LIMIT):
        for j in xrange(TEST_LIMIT):

            n = 4*i**2 + j**2
            if n <= LIMIT and (not (n%12 - 1) or not (n%12 - 5)):
                try:
                    SIEVE[n] = not SIEVE[n]
                except KeyError:
                    pass
                
            n = 3*i**2 + j**2
            if n <= LIMIT and (not(n%12 - 7)):
                try:
                    SIEVE[n] = not SIEVE[n]
                except KeyError:
                    pass

            n = 3*i**2 - j**2
            if n <= LIMIT and i>j and not (n%12 - 11):
                try:
                    SIEVE[n] = not SIEVE[n]
                except KeyError:
                    pass

    for i in xrange(5, TEST_LIMIT):
        try:
            SIEVE[i]
            k = i**2
            for j in xrange(k, LIMIT, k):
                try:
                    if SIEVE[j]: SIEVE[j] = False
                except KeyError:
                    pass
        except KeyError:
            pass
                
    return [2,3] + filter(lambda x: SIEVE[x], xrange(5, LIMIT+1))

def Fibonacci(n):
    "Standard Fibonacci nth number."
    i=0;j=0;k=1;
    for _ in xrange(n):
        i=j;j=k;k=i+j;
    return k

def Nibonacci(i, n=2):
    "Standard Nibonacci nth number"
    pool = [0]*(n-1) + [1]
    for _ in xrange(i):
        pool.append(sum(pool[-i] for i in xrange(1,n+1)))
    return pool[-1]

def rFibonacci(n):
    "Recursive Fibonacci nth number."
    if n <= 1:
        return 1
    else:
        return rFibonacci(n-1) + rFibonacci(n-2)

def rNibonacci(n, i=2):
    "Recursive Nibonacci nth number."
    if i < 2: raise ValueError
    if n < i:
        return 1
    else:
        return sum(rNibonacci(n-(j+1),i) for j in xrange(i))

def product(x):
    return reduce(lambda x,y: x*y, x)

def nCombinations(n,r):
    return Factorial(n)/(Factorial(r)*Factorial(n-r))

def factorial_trailing_zeros(n):
    "trailing_zeros(n) --> n! trailing zeros."
    y=1;z=0;w=1;
    while w:
        w = n/5**y
        z += w
        y += 1
    return z

def trailing_zeros(n, b=10):
    "trailing_zeros(n,b) --> n in base b trailing zeros."
    y=1;
    while not n%b**y:
        y+=1
    return y-1

def flip(x):
    "flip(0) --> 1\nflip(1) --> 0\nflip(True) --> False\nflip(False) --> True"
    return int(not x) if type(x) is int else bool(not x)

def ruler(x):
    e=1;y=2;
    while y<=x:
        if not x%y:
            e+=1;y*=2;
        else:
            break
    return e

def xgray_code(bits=2):
    "Gray Code generator."
    return iter(gray_code(bits));

def gray_code(bits=2):
    "Gray Code Array."
    j = 2**bits
    pool = [[0 for a in range(bits)] for b in range(j)]
    for i in range(1,j):
        y = -ruler(i)
        pool[i] = pool[i-1][:]
        pool[i][y] = flip(pool[i][y])
    return pool
    
        
def printf(x):
    "prints every element of an iterable."
    for _ in x: print _;
    return None

class Fraction(tuple):

    def __new__(self, a=1, b=1):
        assert b;
        sign  = -1 if ((a<0)^(b<0)) else 1;
        (a,b) = simplify(abs(a), abs(b));
        return super(Fraction, self).__new__(self, (sign*a,b));

    def __init__(self, a=1, b=1):
        return None;

    def __float__(self):
        return float(self[0])/float(self[1]);

    def __repr__(self):
        a = self[0];
        b = (flog(max(map(abs,self)),10) + 1)*'-';
        c = self[1];
        x = ' ' if a < 0 else '';
        return "%d\n%s%s\n%s%d"%(a,x,b,x,c);

    def __invert__(self):
        return Fraction( self[1], self[0]);

    def __neg__(self):
        return Fraction(-self[0], self[1]);

    def __add__(self, other):
        if type(other) is not Fraction:
            other = Fraction(other);
        return Fraction(self[0]*other[1] + self[1]*other[0], self[1]*other[1]);

    def __sub__(self, other):
        return Fraction.__add__(self, -other);

    def __mul__(self, other):
        if type(other) is not Fraction:
            other = Fraction(other);
        return Fraction(self[0]*other[0], self[1]*other[1]);

    def __div__(self, other):
        if type(other) is not Fraction:
            other = Fraction(other);
        return Fraction.__mul__(self, ~other);

    def __pow__(self, x):
        return Fraction(self[0]**x, self[1]**x);

    def __rpow__(self, x):
        return (x**self[0])**(1.0/self[1]);


    __radd__ = __add__;
    __rmul__ = __mul__;

def time(n, f, *args):
    times = [];
    for _ in do(n):
        t = clock();
        x = f(*args);
        t = clock() - t;
        times.append(t);
    return min(times);

def test(n, f, *args):
    return f(*args), time(n, f, *args);

def run(main, problem=1):
    init_time = clock()
    answer    = main()
    exec_time = clock() - init_time;

    t_log = log10(exec_time);

    if t_log > 0.0:
        multiplier = 1.0;
        unity = '';
    elif t_log > -3.0:
        multiplier = 1.0e+3;
        unity = 'm';
    else:
        multiplier = 1.0e+6;
        unity = 'u';

    print u"-*- Euler nº%d -*-" % problem 
    print u"The answer is %d."  % answer
    print u"Execution time: %.4f%ss" %(exec_time*multiplier, unity)
    raw_input(u"Press Return to exit.")

    return None;

def next_problem(problem):
    n_problem = problem + 1
    
    with open("euler_template.py","r") as FILE:
        template = FILE.read()
        
    euler = template.replace("problem = 0", "problem = %d" % n_problem, 1)

    try:
        open("euler%s.py" % str(n_problem).zfill(3), "r").close()
        return 1
    except:
        with open("euler%s.py" % str(n_problem).zfill(3), "w") as FILE:
            FILE.write(euler)
        return 0
