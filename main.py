# =========================
# Import Statements
# =========================

from dataclasses import dataclass, field
from drafter import *
from fractions import Fraction
import random
import numpy as np
import matplotlib.pyplot as plt
import re
import math

# =========================
# Dataclasses
# =========================

@dataclass
class Path:
    y_found: bool = False
    x_found: bool = False
    degree_found: bool = False
    factor_found: bool = False
    expand_found: bool = False
    manual_mode: bool = False


@dataclass
class State:
    equation: str = ""
    message: str = ""
    correct_form: str = ""

    # Preserve the user's original input form
    original_form: str = ""

    degree: int = 2

    coefficients: list = field(default_factory=list)
    roots: list = field(default_factory=list)

    y_intercept: int = 0
    x_intercepts: list = field(default_factory=list)

    factors: list = field(default_factory=list)

    path: Path = field(default_factory=Path)

# =========================
# Helper Functions
# =========================

def normalize_factors(expr):
    expr = expr.replace(" ", "")

    factors = re.findall(r'\([^()]+\)', expr)

    # Make sure the entire answer consists of factors.
    if not factors:
        return None

    if "".join(factors) != expr:
        return None

    factors.sort()

    return "".join(factors)


def superscript(power):
    superscripts = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    }

    return "".join(superscripts[digit] for digit in str(power))


def valid_polynomial(expression):
    expression = expression.replace(" ", "")

    if not expression:
        return False

    if "x" not in expression.lower():
        return False

    return True


def parse_standard_polynomial(expression):
    expression = expression.replace(" ", "")

    # Convert superscript exponents into regular exponents.
    superscript_to_caret = {
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4"
    }

    for symbol, replacement in superscript_to_caret.items():
        expression = expression.replace(symbol, replacement)

    if expression == "":
        raise ValueError("Enter a polynomial.")

    # Separate the expression into signed terms.
    terms = re.findall(r"[+-]?[^+-]+", expression)

    # Make sure every character belongs to a term.
    if "".join(terms) != expression:
        raise ValueError("Invalid polynomial.")

    coefficients = {}

    for term in terms:

        if "x" in term:
            match = re.fullmatch(
                r"([+-]?)(\d*)x(?:\^([0-4]))?",
                term
            )

            if match is None:
                raise ValueError("Invalid x term.")

            sign, coefficient_text, power_text = match.groups()

            if coefficient_text == "":
                coefficient = 1
            else:
                coefficient = int(coefficient_text)

            if sign == "-":
                coefficient = -coefficient

            if power_text is None:
                power = 1
            else:
                power = int(power_text)

        else:
            if re.fullmatch(r"[+-]?\d+", term) is None:
                raise ValueError("Invalid constant.")

            coefficient = int(term)
            power = 0

        if power in coefficients:
            coefficients[power] += coefficient
        else:
            coefficients[power] = coefficient

    # Remove any zero coefficients at the beginning.
    while (
        coefficients
        and max(coefficients) > 0
        and coefficients[max(coefficients)] == 0
    ):
        del coefficients[max(coefficients)]

    if not coefficients:
        raise ValueError("The zero polynomial is not supported.")

    degree = max(coefficients.keys())

    final_coefficients = []

    for power in range(degree, -1, -1):
        final_coefficients.append(
            coefficients.get(power, 0)
        )

    return final_coefficients


def factor_integer_polynomial(coefficients):

    from fractions import Fraction

    coefficients = [int(c) for c in coefficients]

    if len(coefficients) < 2 or coefficients[0] == 0:
        raise ValueError("Invalid polynomial")

    original = coefficients[:]
    factors = []

    while len(coefficients) > 2:

        leading = abs(coefficients[0])
        constant = abs(coefficients[-1])

        # Find possible rational roots
        numerators = [
            i for i in range(1, constant + 1)
            if constant % i == 0
        ] if constant else [0]

        denominators = [
            i for i in range(1, leading + 1)
            if leading % i == 0
        ]

        found = False

        for p in numerators:
            for q in denominators:

                for sign in [1, -1]:

                    root = Fraction(sign * p, q)

                    # Synthetic division
                    quotient = [Fraction(coefficients[0])]

                    for coefficient in coefficients[1:-1]:
                        quotient.append(
                            Fraction(coefficient)
                            + root * quotient[-1]
                        )

                    remainder = (
                        Fraction(coefficients[-1])
                        + root * quotient[-1]
                    )

                    if remainder == 0:

                        a = root.denominator
                        b = -root.numerator

                        gcd = math.gcd(abs(a), abs(b))
                        a //= gcd
                        b //= gcd

                        if a < 0:
                            a = -a
                            b = -b

                        factors.append((a, b))

                        coefficients = [
                            int(value) for value in quotient
                        ]

                        found = True
                        break

                if found:
                    break

            if found:
                break

        if not found:
            raise ValueError(
                "This polynomial does not have the supported "
                "rational linear factors."
            )

    # Finish with the remaining linear factor
    a, b = coefficients

    gcd = math.gcd(abs(a), abs(b))

    a //= gcd
    b //= gcd

    if a < 0:
        a = -a
        b = -b

    factors.append((a, b))

    # Verify that the factors reproduce the original polynomial
    reconstructed = np.array([1])

    for a, b in factors:
        reconstructed = np.polymul(reconstructed, [a, b])

    if list(reconstructed) != original:
        raise ValueError(
            "This polynomial cannot be represented "
            "by the supported integer factors."
        )

    return factors


def parse_factored_polynomial(expression):

    expression = expression.replace(" ", "")

    matches = re.findall(
        r"\(([-+]?\d*)x([-+]\d+)\)",
        expression
    )

    factors = []
    roots = []

    polynomial = np.array([1])

    for a_str, b_str in matches:

        if a_str in ("", "+"):
            a = 1
        elif a_str == "-":
            a = -1
        else:
            a = int(a_str)

        b = int(b_str)

        factors.append((a, b))

        polynomial = np.polymul(
            polynomial,
            [a, b]
        )

        roots.append(-b / a)

    coefficients = [int(c) for c in polynomial]

    return (
        coefficients,
        factors,
        roots
    )


def generate_polynomial(state):

    # Random equations must be quadratic, cubic, or quartic.
    degree = random.randint(2, 4)

    state.degree = degree

    factors = []
    roots = []

    polynomial = np.array([1])   # IMPORTANT

    for _ in range(degree):

        while True:
            factor_a = random.choice([1,2,3])
            factor_b = random.choice(
            [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
            )

            if math.gcd(abs(factor_a), abs(factor_b)) == 1:
                break

        factors.append((factor_a, factor_b))

        polynomial = np.polymul(
            polynomial,
            [factor_a, factor_b]
        )

        roots.append(-factor_b / factor_a)

    coefficients = [int(c) for c in polynomial]
    
    state.factors = factors
    state.coefficients = coefficients
    state.roots = roots
    state.y_intercept = coefficients[-1]

    form = random.choice(["standard", "factored"])
    state.original_form = form

    if form == "standard":
        state.correct_form = "standard"
        state.equation = build_standard_form(coefficients)

    else:
        state.correct_form = "factored"
        state.equation = build_factored_form(factors)

    return state.equation


def find_y_intercept(coefficients):
    return coefficients[-1]


def build_standard_form(coefficients):

    degree = len(coefficients) - 1

    result = ""

    for i, coeff in enumerate(coefficients):

        power = degree - i

        if coeff == 0:
            continue

        if coeff > 0 and result:
            result += "+"

        # constant term
        if power == 0:
            result += str(coeff)

        # x term
        elif power == 1:

            if coeff == 1:
                result += "x"

            elif coeff == -1:
                result += "-x"

            else:
                result += f"{coeff}x"

        # higher powers
        else:

            if coeff == 1:
                result += f"x{superscript(power)}"

            elif coeff == -1:
                result += f"-x{superscript(power)}"

            else:
                result += f"{coeff}x{superscript(power)}"

    return result


def build_factored_form(factors):

    result = ""

    for a, b in factors:

        if b > 0:
            result += f"({a}x+{b})"

        else:
            result += f"({a}x{b})"

    return result



def skills_complete(state):

    if state.original_form == "standard":
        form_skill_complete = state.path.factor_found
    else:
        form_skill_complete = state.path.expand_found

    return (
        state.path.y_found
        and state.path.x_found
        and state.path.degree_found
        and form_skill_complete
    )

# =========================
# Pages/Routes
# =========================

@route
def index(state):
    return Page(
        state,
        [
            "Welcome to MathGraph!\nChoose an option:",
            Button("Input Equation", input_page),
            Button("Random Equation", random_equation)
        ]
    )


@route
def input_page(state):
    return Page(
        state,
        [
            "Enter a simple polynomial:\n",
            "Note: for exponents use the ^ character\n",
            "ex. x^2+x+1\n",
            TextBox("equation"),
            Button("Submit", check_equation),
            Button("Home", new_equation)
        ]
    )


@route
def check_equation(state, equation):

    equation = equation.replace(" ", "")
    state.equation = equation

    if not valid_polynomial(equation):
        return Page(
            state,
            [
                "Invalid polynomial.",
                "Please enter a valid polynomial.",
                Button("Try Again", input_page)
            ]
        )

    try:

        if "(" in equation:

            # FACTORED FORM
            state.original_form = "factored"
            state.correct_form = "factored"

            coefficients, factors, roots = (
                parse_factored_polynomial(equation)
            )

            # Make sure the entire expression was parsed
            parsed_expression = "".join(
                re.findall(r"\([^()]+\)", equation)
            )

            if parsed_expression != equation:
                raise ValueError("Invalid factored expression")

            state.coefficients = coefficients
            state.factors = factors
            state.roots = roots

            state.degree = len(factors)

            # Only accept factored polynomials of degree 2 through 4.
            if state.degree not in (2, 3, 4):
                raise ValueError(
                    "Only polynomials of degree 2, 3, or 4 are supported."
                )

        else:

            # STANDARD FORM
            state.original_form = "standard"
            state.correct_form = "standard"

            state.coefficients = parse_standard_polynomial(
                equation
            )

            state.degree = len(state.coefficients) - 1

            # Reject constants and linear equations. The supported
            # polynomial degrees are exactly 2, 3, and 4.
            if state.degree not in (2, 3, 4):
                raise ValueError(
                    "Only polynomials of degree 2, 3, or 4 are supported."
                )

            # Check that the polynomial can be factored
            state.factors = factor_integer_polynomial(
                state.coefficients
            )

            state.roots = [
                -b / a for a, b in state.factors
            ]

        state.y_intercept = state.coefficients[-1]

        # Reset completed skills
        state.path = Path(manual_mode=True)

        state.x_intercepts = []

        return skill_dashboard(state)

    except (ValueError, TypeError, ZeroDivisionError):
        return Page(
            state,
            [
                "Unsupported or invalid polynomial.",
                "Enter a polynomial that can be factored "
                "into integer linear factors.",
                Button("Try Again", input_page),
                Button("Home", new_equation)
            ]
        )


@route
def random_equation(state):
    # Random equations use the same skills-dashboard workflow as
    # manually entered equations, rather than the form-identification flow.
    state.path = Path(manual_mode=True)
    state.equation = generate_polynomial(state)
    state.x_intercepts = []

    return skill_dashboard(state)


@route
def new_equation(state):

    state.equation = ""
    state.correct_form = ""
    state.original_form = ""

    state.degree = 2

    state.coefficients = []
    state.roots = []
    state.factors = []

    state.y_intercept = 0
    state.x_intercepts = []

    state.path = Path()

    return index(state)


@route
def skill_dashboard(state):

    items = [
        "Choose a Skill",
        "\n",
        f"Equation: {state.equation}",
        "\n"
    ]

    if state.original_form == "standard":

        # Y-INTERCEPT
        if state.path.y_found:
            items.append(
                f"Y-Intercept: (0, {state.y_intercept})"
            )
        else:
            items.append(
                Button("Find Y-Intercept", y_intercept_instructions)
            )

        items.append("\n")

        # DEGREE
        if state.path.degree_found:
            items.append(f"Degree: {state.degree}")
        else:
            items.append(
                Button("Find Degree", degree_instructions)
            )

        items.append("\n")

        # FACTOR
        if state.path.factor_found:
            items.append(
                f"Factored Form: {state.equation}"
            )
        else:
            items.append(
                Button("Factor", factor_standard_instructions)
            )

        items.append("\n")

        # X-INTERCEPTS: available after factoring
        if state.path.factor_found:

            if state.path.x_found:
                items.append(
                    f"X-Intercepts: {state.x_intercepts}"
                )
            else:
                items.append(
                    Button(
                        "Find X-Intercepts",
                        x_intercept_instructions
                    )
                )

    elif state.original_form == "factored":

        # X-INTERCEPTS
        if state.path.x_found:
            items.append(
                f"X-Intercepts: {state.x_intercepts}"
            )
        else:
            items.append(
                Button(
                    "Find X-Intercepts",
                    x_intercept_instructions
                )
            )

        items.append("\n")

        # EXPAND
        if state.path.expand_found:
            items.append(
                f"Standard Form: {state.equation}"
            )
        else:
            items.append(
                Button(
                    "Expand",
                    expand_factored_instructions
                )
            )

        items.append("\n")

        # Y-INTERCEPT AND DEGREE:
        # Only available after expansion
        if state.path.expand_found:

            if state.path.y_found:
                items.append(
                    f"Y-Intercept: (0, {state.y_intercept})"
                )
            else:
                items.append(
                    Button(
                        "Find Y-Intercept",
                        y_intercept_instructions
                    )
                )

            items.append("\n")

            if state.path.degree_found:
                items.append(f"Degree: {state.degree}")
            else:
                items.append(
                    Button("Find Degree", degree_instructions)
                )

    items.append("\n")

    # FINAL GRAPH BUTTON
    if (
        state.path.y_found
        and state.path.x_found
        and state.path.degree_found
        and (
            state.path.factor_found
            if state.original_form == "standard"
            else state.path.expand_found
        )
    ):
        items.append(
            Button("Graph My Equation", final_graph)
        )

    items.append("\n")
    items.append(Button("Home", new_equation))

    return Page(state, items)


@route
def form_instructions(state):
    return Page(
        state,
        [
            "Standard Form vs. Factored Form",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back", index),
            Button("Next", identify_form)
        ]
    )


@route
def identify_form(state):
    return Page(
        state,
        [
            "Equation:",
            state.equation,
            "\nIs this equation in Standard Form or Factored Form?\n",
            Button("Standard Form", choose_standard),
            Button("Factored Form", choose_factored),
            "\n",
            Button("Home", new_equation),
            Button("Back", form_instructions)
        ]
    )


@route
def choose_standard(state):
    
    if state.correct_form == "standard":
        return continue_page(state)

    return Page(
        state,
        [
            "Try again. This equation is not in Standard Form.",
            "\n",
            Button("Home", new_equation),
            Button("Back", identify_form)
        ]
    )


@route
def choose_factored(state):

    if state.correct_form == "factored":
        return continue_page(state)

    return Page(
        state,
        [
            "Try again. This equation is not in Factored Form.",
            "\n",
            Button("Home", new_equation),
            Button("Back", identify_form)
        ]
    )

@route
def continue_page(state):

    if state.correct_form == "standard":
        return y_intercept_instructions(state)

    return x_intercept_instructions(state)


@route
def y_intercept_instructions(state):
    back_page = (
        skill_dashboard
        if state.path.manual_mode
        else identify_form
    )
    return Page(
        state,
        [
            "How to Find the Y-Intercept",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back", back_page),
            Button("Next", y_intercept_page)
        ]
    )


@route
def y_intercept_page(state):
    return Page(
        state,
        [
            "Find the y-intercept\n"
            "Equation:",
            state.equation,
            "\n",
            TextBox("answer"),
            Button("Submit", check_y_intercept),
            "\n",
            Button("Home", new_equation),
            Button("Back", y_intercept_instructions)
        ]
    )


@route
def check_y_intercept(state, answer):

    correct = find_y_intercept(state.coefficients)

    try:
        answer = int(answer)
    except:
        return Page(
            state,
            [
                "Please enter a number.",
                Button("Back", y_intercept_instructions)
            ]
        )

    if answer != correct:
        return Page(
            state,
            [
                "Incorrect ",
                Button("Try Again", y_intercept_page)
            ]
        )

    state.y_intercept = correct
    state.path.y_found = True

    return graph_y_intercept(state)


@route
def graph_y_intercept(state):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Y-intercept
    ax.scatter(
        [0],
        [state.y_intercept],
        color="red",
        s=150,
        label="Y-Intercept"
    )

    # X-intercepts if already found
    if state.path.x_found:
        ax.scatter(
            state.x_intercepts,
            [0] * len(state.x_intercepts),
            color="green",
            s=150,
            label="X-Intercepts"
        )

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")

    if state.x_intercepts:
        smallest = min(state.x_intercepts)
        largest = max(state.x_intercepts)

        ax.set_xlim(
            min(-10, smallest - 2),
            max(10, largest + 2)
        )
    else:
        ax.set_xlim(-10, 10)

    ax.set_ylim(
        min(-10, state.y_intercept - 2),
        max(10, state.y_intercept + 2)
    )

    ax.grid(True)
    ax.legend()
    ax.set_title("Intercepts")

    if state.path.manual_mode:
        return Page(
            state,
            [
                "Correct!",
                f"Y-Intercept = (0, {state.y_intercept})",
                MatPlotLibPlot(),
                Button("Continue", skill_dashboard)
            ]
        )

    if state.path.x_found:
        next_route = degree_instructions
    else:
        next_route = factor_standard_instructions
    
    return Page(
        state,
        [
            "Correct!",
            f"Y-Intercept = (0,{state.y_intercept})",
            MatPlotLibPlot(),
            Button("Continue", next_route)
        ]
    )


@route
def x_intercept_instructions(state):
    return Page(
        state,
        [
            "How to Find X-Intercepts",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back",
                skill_dashboard
                if state.path.manual_mode
                else identify_form
            ),
            Button("Next", x_intercept_page)
        ]
    )


@route
def x_intercept_page(state):

    return Page(
        state,
        [
            "Find the X-Intercepts",
            "\nEquation:",
            state.equation,
            "\nNote: enter values separated by commas\n",
            "ex. -2,3",
            "Enter non whole number roots as fractions or decimals rounded to two places\n",
            "ex. 1/3 or 0.33\n",
            TextBox("answer"),
            Button("Submit", check_x_intercepts),
            "\n",
            Button("Home", new_equation),
            Button("Back", x_intercept_instructions)
        ]
    )


@route
def check_x_intercepts(state, answer):

    try:
        user_roots = [
            round(float(Fraction(value.strip())), 2)
            for value in answer.split(",")
        ]

    except:
        return Page(
            state,
            [
                "Invalid Input",
                "Enter roots separated by commas.",
                Button("Back", x_intercept_instructions)
            ]
        )

    correct_roots = sorted(
        list(
            set(
                round(root, 2)
                for root in state.roots
            )
        )
    )

    user_roots = sorted(
        list(
            set(user_roots)
        )
    )

    if len(user_roots) != len(correct_roots):
        return Page(
            state,
            [
                "Incorrect",
                Button("Try Again", x_intercept_page)
            ]
        )

    for user, correct in zip(user_roots, correct_roots):
        if abs(user - correct) > 0.01:
            return Page(
                state,
                [
                    "Incorrect",
                    Button("Try Again", x_intercept_page)
                ]
            )

    state.x_intercepts = correct_roots
    state.path.x_found = True

    return graph_x_intercepts(state)


@route
def graph_x_intercepts(state):

    fig, ax = plt.subplots(figsize=(6, 6))

    # X-intercepts
    ax.scatter(
        state.x_intercepts,
        [0] * len(state.x_intercepts),
        color="green",
        s=150,
        label="X-Intercepts"
    )

    # Y-intercept if already found
    if state.path.y_found:
        ax.scatter(
            [0],
            [state.y_intercept],
            color="red",
            s=150,
            label="Y-Intercept"
        )

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")

    if state.x_intercepts:
        smallest = min(state.x_intercepts)
        largest = max(state.x_intercepts)

        ax.set_xlim(
            min(-10, smallest - 2),
            max(10, largest + 2)
        )
    else:
        ax.set_xlim(-10, 10)

    # Scale graph based on y-intercept if it exists
    if state.path.y_found:
        ax.set_ylim(
            min(-10, state.y_intercept - 2),
            max(10, state.y_intercept + 2)
        )
    else:
        ax.set_ylim(-10, 10)

    ax.grid(True)
    ax.legend()
    ax.set_title("Intercepts")

    if state.path.manual_mode:
        return Page(
            state,
            [
                "Correct!",
                f"X-Intercepts = {[(x, 0) for x in state.x_intercepts]}",
                MatPlotLibPlot(),
                Button("Continue", skill_dashboard)
            ]
        )

    if state.path.y_found:
        next_route = degree_instructions
    else:
        next_route = expand_factored_instructions
    
    return Page(
        state,
        [
            "Correct!",
            f"X-Intercepts = {[(x,0) for x in state.x_intercepts]}",
            MatPlotLibPlot(),
            Button("Continue", next_route)
        ]
    )


@route
def factor_standard_instructions(state):
    return Page(
        state,
        [
            "How to Convert to Factored Form",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back", skill_dashboard if state.path.manual_mode else graph_y_intercept),
            Button("Next", factor_standard_page)
        ]
    )


@route
def factor_standard_page(state):

    return Page(
        state,
        [
            "Convert to Factored Form",
            "\nEquation:",
            state.equation,
            "\nNote: include the 1 coeffecient in your answer if applicable\n",
            "ex. (1x-2)(1x-3)\n",
            TextBox("answer"),
            Button("Submit", check_factored_form),
            Button("Back", factor_standard_instructions),
            Button("Home", new_equation)
        ]
    )


@route
def check_factored_form(state, answer):

    answer = answer.replace(" ", "")

    correct = build_factored_form(state.factors)

    user_factors = normalize_factors(answer)
    correct_factors = normalize_factors(correct)

    if user_factors is None or user_factors != correct_factors:

        return Page(
            state,
            [
                "Incorrect",
                Button("Try Again", factor_standard_page),
                Button("Back", skill_dashboard)
            ]
        )

    state.equation = correct
    state.correct_form = "factored"

    state.path.factor_found = True

    return skill_dashboard(state)


@route
def expand_factored_instructions(state):
    return Page(
        state,
        [
            "How to Convert to Standard Form",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back", skill_dashboard if state.path.manual_mode else graph_x_intercepts),
            Button("Next", expand_factored_page)
        ]
    )


@route
def expand_factored_page(state):

    return Page(
        state,
        [
            "Convert to Standard Form",
            "\nEquation:",
            state.equation,
            "\nNote: for exponents use the ^ character\n",
            "ex. x^2+x+1\n",
            TextBox("answer"),
            Button("Submit", check_standard_form),
            "\n",
            Button("Home", new_equation),
            Button("Back", expand_factored_instructions)
        ]
    )


@route
def check_standard_form(state, answer):

    answer = answer.replace(" ", "")
    answer = answer.replace("^2", "²")
    answer = answer.replace("^3", "³")
    answer = answer.replace("^4", "⁴")

    correct = build_standard_form(
        state.coefficients
    ).replace(" ", "")

    if answer != correct:

        return Page(
            state,
            [
                "Incorrect ",
                Button("Try Again", expand_factored_page),
                Button("Back", skill_dashboard)
            ]
        )

    state.equation = correct
    state.correct_form = "standard"

    state.path.expand_found = True

    return skill_dashboard(state)


@route
def degree_instructions(state):
    return Page(
        state,
        [
            "How to Find the Degree of a Polynomial",

            "\nInput Instructions Here\n",

            Button("Home", new_equation),
            Button("Back",
                skill_dashboard
                if state.path.manual_mode
                else graph_y_intercept
            ),
            Button("Next", degree_page)
        ]
    )


@route
def degree_page(state):
    
    if state.path.manual_mode:
        standard_form = state.equation
    else:
        standard_form = build_standard_form(
            state.coefficients
        )

    return Page(
        state,
        [
            "Equation:",
            standard_form,

            "\nWhat is the degree of your polynomial?",
            TextBox("degree_answer"),

            "\nIs the degree Odd or Even?",
            TextBox("parity_answer"),

            Button("Submit", check_degree),
            "\n",
            Button("Home", new_equation),
            Button("Back", degree_instructions)
        ]
    )


@route
def check_degree(state, degree_answer, parity_answer):

    # Check degree is a valid integer
    try:
        degree_answer = int(degree_answer)
    except:
        return Page(
            state,
            [
                "Incorrect\n",
                "Degree must be a valid number.",
                Button("Try Again", degree_page)
            ]
        )

    parity_answer = parity_answer.strip().lower()

    correct_parity = "even" if state.degree % 2 == 0 else "odd"

    if degree_answer != state.degree or parity_answer != correct_parity:
        return Page(
            state,
            [
                "Incorrect ",
                Button("Try Again", degree_page)
            ]
        )
    
    
    state.path.degree_found = True

    if state.path.manual_mode:
        return skill_dashboard(state)

    if skills_complete(state):
        return summary_page(state)

    return skill_dashboard(state)




@route
def summary_page(state):
    return Page(
        state,
        [

            f"Equation: {build_standard_form(state.coefficients)}",

            "\n",
            f"Degree: {state.degree}",

            "\n",
            f"Y-Intercept: (0, {state.y_intercept})",

            "\n",
            f"X-Intercepts: {[(x, 0) for x in state.x_intercepts]}",

            "\n",
            Button("Home", new_equation),
            Button("Graph My Equation", final_graph)
        ]
    )



@route
def final_graph(state):

    if state.roots:
        smallest = min(state.roots)
        largest = max(state.roots)

        x_min = min(-10, smallest - 2)
        x_max = max(10, largest + 2)
    else:
        x_min = -10
        x_max = 10

    x = np.linspace(x_min, x_max, 500)

    y = np.polyval(
        state.coefficients,
        x
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x, y, color="blue")

    # Plot y-intercept
    ax.scatter(
        [0],
        [state.y_intercept],
        color="red",
        s=100
    )

    # Plot x-intercepts
    ax.scatter(
        state.x_intercepts,
        [0] * len(state.x_intercepts),
        color="green",
        s=100
    )

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.grid(True)
    ax.set_title("Final Polynomial Graph")

    return Page(
        state,
        [
            "Final Graph",
            f"Equation: {build_standard_form(state.coefficients)}",
            MatPlotLibPlot(),
            Button("Home", new_equation)
        ]
    )

# =========================
# Start Program
# =========================

start_server(State())
