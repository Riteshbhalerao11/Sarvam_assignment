from typing import List, Set, Union
from .errors import EinopsError

class AnonymousAxis:
    """
    Anonymous axis representation ensuring unique instances.
    
    """
    def __init__(self, value: str):
        self.value = int(value)
        if self.value < 1:
            raise EinopsError(f"Anonymous axis should have positive length, not {self.value}")
    
    def __repr__(self):
        return "{}-axis".format(str(self.value))

class Parser:
    """
    Parser to break down pattern into suitable format required for further operations.

    """
    def __init__(self, pattern: str, is_input: bool):

        self.has_ellipsis: bool = "..." in pattern
        self.identifiers: Set[str] = set()
        self.structure: List[Union[List[str], str]] = []
        
        # Validate ellipsis presence
        if "." in pattern:
            if "..." not in pattern or pattern.count("...") != 1 or pattern.count(".") != 3:
                raise EinopsError("Expression may contain dots only inside ellipsis (...) and only once.")
        
        bracket_group = None
        tokens = pattern.replace("(", " ( ").replace(")", " ) ").split()
        
        # Bracket counts
        lb = 0 
        rb = 0

        for token in tokens:
            if token == "(":
                lb+=1
                bracket_group = []
            elif token == ")":
                rb+=1
                if bracket_group is None:
                    raise EinopsError(f"Mismatched parentheses in pattern : {pattern}")
                self.structure.append(bracket_group)
                bracket_group = None
            else:
                self.add_axis(token, bracket_group, is_input, pattern)
        if lb >  rb:
            raise EinopsError(f"Paranthesis are not properly closed in pattern : {pattern}")

    def add_axis(self, token: str, bracket_group: list, is_input: bool, pattern: str):
        """
        Handles axis addition, ensuring uniqueness and validity.
        
        """
        if token in self.identifiers:
            raise EinopsError(f'Pattern contains duplicate dimension "{token}"')

        if token == "...":
            if bracket_group is None:
                self.structure.append("ELLIPSIS") # Proxy for "..."
            else:
                if is_input:
                    raise EinopsError(f"Ellipsis inside parentheses on the left side is not allowed: {pattern}")
                bracket_group.append("ELLIPSIS")
            self.identifiers.add("ELLIPSIS")

        else:

            is_number = token.isdecimal()

            # Handles singleton dimension
            if is_number and int(token) == 1:

                # Accounts for decompositon of axis
                if bracket_group is None:
                    self.structure.append([])
                return
            
            # Handles anonymous axes
            if is_number:
                if is_input:
                    raise EinopsError(f"Anonymous axis (except 1) in the left side is not allowed: {pattern}")
                token = AnonymousAxis(token)
            
            # Handles normal (str) axes 
            else:
                if not str.isidentifier(token):
                    raise EinopsError(f"Invalid identifier for axis name. Only valid python identifiers are allowed. : {token}")
            
            self.identifiers.add(token)

            if bracket_group is None:
                self.structure.append([token])
            else:
                bracket_group.append(token)
