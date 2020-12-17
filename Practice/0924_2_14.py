#!/usr/bin/python3

def Test_Lists():
    xs = [3, 1, 2]
    print(xs, xs[2])
    print(xs[-1])
    xs[2] = "foo"
    print(xs)
    xs.append("bar")
    print(xs)
    X = xs.pop()
    print(X, xs)

if __name__ == '__main__':
    Test_Lists()