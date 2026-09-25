"""Minimal reverse-mode autodiff, pure stdlib.

Only exists so a real language model can be trained in this repo without a
numeric stack. Correctness is checked against finite differences in
tests.py rather than assumed, because a silently wrong gradient trains a
model that looks plausible and has learned nothing.
"""
import math


class V:
    __slots__ = ("d", "g", "_back", "_prev")

    def __init__(self, d, prev=(), back=None):
        self.d = d
        self.g = 0.0
        self._prev = prev
        self._back = back

    def __add__(self, o):
        o = o if isinstance(o, V) else V(o)
        out = V(self.d + o.d, (self, o))

        def back():
            self.g += out.g
            o.g += out.g
        out._back = back
        return out

    def __mul__(self, o):
        o = o if isinstance(o, V) else V(o)
        out = V(self.d * o.d, (self, o))

        def back():
            self.g += o.d * out.g
            o.g += self.d * out.g
        out._back = back
        return out

    def __pow__(self, k):
        out = V(self.d ** k, (self,))

        def back():
            self.g += k * (self.d ** (k - 1)) * out.g
        out._back = back
        return out

    def relu(self):
        out = V(self.d if self.d > 0 else 0.0, (self,))

        def back():
            self.g += (out.d > 0) * out.g
        out._back = back
        return out

    def exp(self):
        out = V(math.exp(min(self.d, 60.0)), (self,))

        def back():
            self.g += out.d * out.g
        out._back = back
        return out

    def log(self):
        out = V(math.log(max(self.d, 1e-12)), (self,))

        def back():
            self.g += (1.0 / max(self.d, 1e-12)) * out.g
        out._back = back
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, V) else V(o)))

    def __truediv__(self, o):
        return self * ((o if isinstance(o, V) else V(o)) ** -1)

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        order, seen = [], set()

        def build(v):
            # Iterative: a recursive walk blows the stack on a graph this
            # deep, and the depth grows with sequence length.
            stack = [(v, False)]
            while stack:
                n, done = stack.pop()
                if done:
                    order.append(n)
                    continue
                if id(n) in seen:
                    continue
                seen.add(id(n))
                stack.append((n, True))
                for p in n._prev:
                    if id(p) not in seen:
                        stack.append((p, False))
        build(self)
        self.g = 1.0
        for n in reversed(order):
            if n._back:
                n._back()
