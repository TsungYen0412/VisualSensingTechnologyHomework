#!/usr/bin/python3

class Foo():
    __aoo = 123
    
    def __boo(self):
        return 123
    
    def coo(self):
        return 456

if __name__ == '__main__':
    f = Foo()
    print(dir(f))
    print(f._Foo__boo())
    print(f.__boo())