# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if __name__ == '__main__':
    print(is_interactive())
