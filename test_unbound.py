def foo(x):
    return x + 1

def bar():
    foo = foo(1)
    return foo

bar()
