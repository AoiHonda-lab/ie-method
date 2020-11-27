class Hyperreals(object):
    def __init__(self, number):
        import sys
        maxInt = sys.maxsize 
        if number == "MAX_INFINITY" or number >= maxInt:
            self.value = "MAX_INFINITY"
        else:
            if number == "MIN_INFINITY" or number <= -1 * maxInt -1:
                self.value = "MIN_INFINITY"
            else:
                self.value = number
    def ge(self, otherNumber):
        if self.value == "MAX_INFINITY":
            if otherNumber.value == "MAX_INFINITY":
                raise ValueError("Infinity comparison")
            else:
                return True
        else:
            if self.value == "MIN_INFINITY":
                if self.value == "MIN_INFINITY":
                    raise ValueError("Infinity comparison")
                else:
                    return False
            else:
                if otherNumber.value == "MAX_INFINITY":
                    return False
                else:
                    if otherNumber.value == "MIN_INFINITY":
                        return True
                    else:
                        return self.value >= otherNumber.value
    def add(self, otherNumber):
        if self.value == "MAX_INFINITY" and otherNumber.value == "MIN_INFINITY" or otherNumber.value == "MAX_INFINITY" and self.value == "MIN_INFINITY":
            raise ValueError("Sum of opposite infinities")
        else:
            if self.value == "MAX_INFINITY" or otherNumber.value == "MAX_INFINITY":
                return Hyperreals("MAX_INFINITY")
            else:
                if self.value == "MIN_INFINITY" or otherNumber.value == "MIN_INFINITY":
                    return Hyperreals("MIN_INFINITY")
                else:
                    return Hyperreals(self.value + otherNumber.value)