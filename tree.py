from eulertools import *;
    
class Arvore(dict):

    def __init__(self, n):
        dict.__init__(self,{});
        
        tree = [[()]]+[list(combinations(range(1,n+1),i))
                       for i in range(n,0,-1)][::-1];

        for i in range(len(tree)-1):
            for j in range(len(tree[i])):
                key = tree[i][j];
                l   = len(key);
                for k in range(len(tree[i+1])):
                    item = tree[i+1][k];
                    if item[:l] == key:
                        self[item] = key;
                    else:
                        continue;

        return None;

    def __repr__(self):
        items  = sorted(self.items());
        string = "Root\n";
        string+= "\n".join(["%r -> %r" % tup for tup in items]);
        return string.replace("(","{").replace(")","}");
            
