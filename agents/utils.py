# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício J V Ceolin
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if __name__ == '__main__':
    print(is_interactive())
