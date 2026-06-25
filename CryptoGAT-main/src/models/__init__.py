try:
    from .gat import GAT
    __all__ = ['GAT']
except ImportError:
    __all__ = []
