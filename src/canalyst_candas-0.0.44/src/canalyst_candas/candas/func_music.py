class FuncMusic(object):
    """
    Helper class for on the fly function generation.

    Used in the ForecastFrame concept where we try to change one param
    across multiple tickers for re-fit in the scenario engine.

    """

    def apply_function(self, value, modifier, argument):
        """
        Apply a specified function to a value

        Parameters
        ----------
        value: int
        modifier: str
        {"add", "subtract", "divide", "multiply","value"}
        argument: int
        """
        self.value = value
        self.modifier = modifier
        self.argument = argument
        method_name = "func_" + str(self.modifier)
        method = getattr(
            self,
            method_name,
            lambda: "Invalid function: use add, subtract, divide, or multiply",
        )
        return method()

    def func_add(self):
        return float(self.value) + float(self.argument)

    def func_divide(self):
        return float(self.value) / float(self.argument)

    def func_multiply(self):
        return float(self.value) * float(self.argument)

    def func_subtract(self):
        return float(self.value) - float(self.argument)
