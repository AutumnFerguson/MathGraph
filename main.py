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
    standard_equation: str = ""
    factored_equation: str = ""
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
        return Page(state, [
            "Invalid polynomial.",
            "Please enter a valid polynomial.",
            Button("Try Again", input_page)
        ])

    try:
        if "(" in equation:
            state.original_form = "factored"
            state.correct_form = "factored"
            coefficients, factors, roots = parse_factored_polynomial(equation)
            parsed_expression = "".join(re.findall(r"\([^()]+\)", equation))
            if parsed_expression != equation or len(factors) not in (2, 3, 4):
                raise ValueError("Invalid factored expression")
            state.coefficients = coefficients
            state.factors = factors
            state.roots = roots
            state.degree = len(factors)
            state.factored_equation = equation
            state.standard_equation = build_standard_form(coefficients)
        else:
            state.original_form = "standard"
            state.correct_form = "standard"
            state.coefficients = parse_standard_polynomial(equation)
            state.degree = len(state.coefficients) - 1
            if state.degree not in (2, 3, 4):
                raise ValueError("Only degrees 2, 3, and 4 are supported.")
            state.factors = factor_integer_polynomial(state.coefficients)
            state.roots = [-b / a for a, b in state.factors]
            state.standard_equation = build_standard_form(state.coefficients)
            state.factored_equation = build_factored_form(state.factors)

        state.y_intercept = state.coefficients[-1]
        state.path = Path(manual_mode=True)
        state.x_intercepts = []
        return form_instructions(state)

    except (ValueError, TypeError, ZeroDivisionError):
        return Page(state, [
            "Unsupported or invalid polynomial.",
            "Enter a polynomial that can be factored into integer linear factors.",
            Button("Try Again", input_page),
            Button("Home", new_equation)
        ])


@route
def random_equation(state):
    state.path = Path(manual_mode=True)
    state.equation = generate_polynomial(state)
    state.x_intercepts = []
    if state.original_form == "standard":
        state.standard_equation = state.equation
        state.factored_equation = build_factored_form(state.factors)
    else:
        state.factored_equation = state.equation
        state.standard_equation = build_standard_form(state.coefficients)
    return form_instructions(state)


@route
def new_equation(state):
    state.equation = ""
    state.standard_equation = ""
    state.factored_equation = ""
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
    standard = state.standard_equation or build_standard_form(state.coefficients)
    factored = state.factored_equation or build_factored_form(state.factors)
    return Page(state, [
        f"Standard Form: {standard}",
        "\n",
        f"Factored Form: {factored}",
        "\n",
        (
            f"X-Intercepts: {[(x, 0) for x in state.x_intercepts]}"
            if state.path.x_found
            else Button("Find X-Intercepts", x_intercept_instructions)
        ),
        "\n",
        (
            f"Y-Intercept: (0, {state.y_intercept})"
            if state.path.y_found
            else Button("Find Y-Intercept", y_intercept_instructions)
        ),
        "\n",
        (
            f"Degree: {state.degree}"
            if state.path.degree_found
            else Button("Find Degree", degree_instructions)
        ),
        "\n",
        Button("Graph My Equation", final_graph)
            if state.path.x_found and state.path.y_found and state.path.degree_found
            else "Complete all three skills to unlock the final graph.",
        "\n",
        Button("Home", new_equation)
    ])


@route
def form_instructions(state):
    return Page(state, [
        "Factored Form or Standard Form?\n",
        "\n",
        "Standard Form: ax² + bx + c\n",
        "Expanded terms written in descending order of power.\n",
        "\n",
        "Factored Form: a(x - p)(x - q)\n",
        "A product of linear factors that shows the roots.\n",
        Button("Home", new_equation),
        Button("Next", identify_form)
    ])


@route
def identify_form(state):
    return Page(state, [
        "Identify the equation's form:\n",
        f"Equation: {state.equation}",
        "\n",
        Button("Standard Form", choose_standard),
        Button("Factored Form", choose_factored),
        "\n",
        Button("Home", new_equation),
        Button("Back", form_instructions)
    ])


@route
def choose_standard(state):
    if state.correct_form != "standard":
        return Page(state, [
            "Try again. This equation is not in Standard Form.",
            Button("Try Again", identify_form),
            Button("Home", new_equation)
        ])
    return factor_standard_instructions(state)


@route
def choose_factored(state):
    if state.correct_form != "factored":
        return Page(state, [
            "Try again. This equation is not in Factored Form.",
            Button("Try Again", identify_form),
            Button("Home", new_equation)
        ])
    return expand_factored_instructions(state)


@route
def continue_page(state):
    if state.correct_form == "standard":
        return factor_standard_instructions(state)
    return expand_factored_instructions(state)


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
            "To find the y-intercept of any polynomial function written in standard form substitute x=0 into the polynomial function and simplify.\n",
            "Since every term with a variable x becomes 0, the y-intercept is always the constant term (c) at the end of the standard form expression.\n",
            "\n",
            "Example: f(x)=3x"+superscript(4)+"-2x"+superscript(3)+"+5x"+superscript(2)+"-7x+9\n",
            "\n",
            "Substitute 0 for every x:\n",
            "f(0)=3(0)"+superscript(4)+"-2(0)"+superscript(3)+"+5(0)"+superscript(2)+"-7(0)+9\n",
            "\n",
            "Simplify the expression:\n",
            "f(0)=0-0+0-0+9=9\n",
            "\n",
            "Y-intercept: (0,9)\n",
            "\n",
            "A Quick Shortcut if the polynomial is already in standard form, you can skip the math and simply look at the last term (the number without an x).\n",
            "For f(x)=2x"+superscript(2)+"+8x-12, the constant is -12, so the y-intercept is (0,-12)\n",
            "For f(x)=x"+superscript(4)+"+2x"+superscript(3)+"-9x"+superscript(2)+"-18x, there is no constant term (it is +0), so the y-intercept is (0,0)\n",

            Button("Home", new_equation),
            Button("Back", back_page),
            Button("Next", y_intercept_page)
        ]
    )


@route
def y_intercept_page(state):
    return Page(state, [
        "Find the y-intercept\n",
        "Equation:",
        state.standard_equation,
        "\n",
        TextBox("answer"),
        Button("Submit", check_y_intercept),
        "\n",
        Button("Home", new_equation),
        Button("Back", y_intercept_instructions)
    ])


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
            "How to Find X-Intercepts\n",
            "To find the x-intercepts of a polynomial function in factored form,\n",
            "set the function equal to zero (f(x)=0) and apply the zero product property by setting each individual factor equal to zero and solving for x.\n",
            "\n",
            "Example: f(x)=(x+2)(x-3)(2x+4)\n",
            "\n",
            "Set the function to 0:\n",
            "(x+2)(x-3)(3x+4)=0\n",
            "\n",
            "Set each factor equal to zero:\n",
            "x+2=0\n",
            "x-3=0\n",
            "3x+4=0\n",
            "\n",
            "Solve for x:\n",
            "x=-2\n",
            "x=3\n",
            "x=-4/3\n",
            "\n",
            "Write as ordered pairs:\n",
            "(-2,0), (3,0), and (-4/3,0)\n",

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
    return Page(state, [
        "Find the X-Intercepts\n",
        "Equation:",
        state.factored_equation,
        "\n",
        "Enter values separated by commas (example: -2,3)\n",
        "Enter non-whole-number roots as fractions or decimals rounded to two places\n",
        TextBox("answer"),
        Button("Submit", check_x_intercepts),
        Button("Home", new_equation),
        Button("Back", x_intercept_instructions)
    ])


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
            "How to factor?\n",
            "\n",
            "Step 1: Always Check for a Greatest Common Factor (GCF)\n",
            "Look at every term in the polynomial and find the largest number and lowest power of any variable that divides evenly into all of them.\n",
            "\n",
            "Example: 4x"+superscript(3)+"-8x"+superscript(2)+"\n",
            "The GCF is 4x"+superscript(2)+".\n",
            "Factored: 4x"+superscript(2)+"(x-2)\n",
            "\n",
            "Step 2: Factor by Number of Terms\n",
            "Count how many terms are left in the expression\n",
            "\n",
            "Binomials (2 Terms)\n",
            "Difference of Squares: If you have two squared terms separated by a minus sign (a"+superscript(2)+"-b"+superscript(2)+"), use the pattern:\n",
            "a"+superscript(2)+"-b"+superscript(2)+"=(a+b)(a-b)\n",
            "\n",
            "Example: x"+superscript(2)+"-9=(x+3)(x-3)\n",
            "\n",
            "Trinomials (3 Terms)\n",
            "When the leading coefficient is 1 like in x"+superscript(2)+"+bx+c you can find two numbers that multiply to c and add up to b\n",
            "\n",
            "Example: x"+superscript(2)+"+5x+6\n",
            "Factors of 6 that add to 5 are 2 and 3 (2*3=6, 2+3=5)\n",
            "Factored form: (x+2)(x+3)\n",
            "\n",
            "When the leading coefficient is not 1 like in ax"+superscript(2)+"+bx+c you slide a over and multiply it by c\n", 
            "x"+superscript(2)+"+bx+ac\n",
            "\n",
            "Example: 2x"+superscript(2)+"+5x+3\n",
            "Slide: x"+superscript(2)+"+5x+3(2)=x"+superscript(2)+"+5x+6\n",
            "Then factor normally by finding two numbers that multiply to c and add to b just like when the leading coefficient is 1\n",
            "Factors of 6 that add to 5 are 2 and 3 (2*3=6, 2+3=5) so (x+2)(x+3)\n",
            "Now divide your new roots by a and simplify\n",
            "(x+2/2)(x+3/2) = (x+1)(x+3/2)\n",
            "Finally bottoms up any denominators to the coefficient of x!\n",
            "Factored Form: (x+1)(2x+3)\n",
            "\n",
            "Four-Term Polynomials\n",
            "Factoring by Grouping: Group the first two terms together and the last two terms together\n",
            "Factor out the GCF from each individual group so that a common binomial remains\n",
            "\n",
            "Example: x"+superscript(3)+"-4x"+superscript(2)+"+3x-12 = x"+superscript(2)+"(x-4)+3(x-4) = (x-4)(x"+superscript(2)+"+3)\n",
            Button("Home", new_equation),
            Button("Back", skill_dashboard if state.path.manual_mode else graph_y_intercept),
            Button("Next", factor_standard_page)
        ]
    )


@route
def factor_standard_page(state):
    return Page(state, [
        "Convert to Factored Form\n",
        "Equation: ",
        state.standard_equation,
        "\n",
        "Note: Include any coeffecients of 1\n",
        "ex. (x-3)(x-2) = (1x-3)(1x-2)\n",
        TextBox("answer"),
        Button("Submit", check_factored_form),
        "\n",
        Button("Back", factor_standard_instructions),
        Button("Home", new_equation)
    ])


@route
def check_factored_form(state, answer):
    answer = answer.replace(" ", "")
    correct = build_factored_form(state.factors)
    user_factors = normalize_factors(answer)
    correct_factors = normalize_factors(correct)
    if user_factors is None or user_factors != correct_factors:
        return Page(state, [
            "Incorrect. Try again.",
            Button("Try Again", factor_standard_page),
            Button("Back", factor_standard_instructions)
        ])
    state.factored_equation = correct
    state.path.factor_found = True
    return skill_dashboard(state)


@route
def expand_factored_instructions(state):
    return Page(
        state,
        [
            "How to expand\n",
            "\n",
            "One way is to use FOIL\n",
            "\n",
            "The word FOIL is an acronym that stands for:\n",
            "First – Multiply the first terms in each binomial\n",
            "Outer – Multiply the outer (outside) terms\n",
            "Inner – Multiply the inner (inside) terms\n",
            "Last – Multiply the last terms in each binomial\n",
            "\n",
            "After multiplying these four pairs, you add the results together and combine any like terms to get your final polynomial\n",
            "\n",
            "Example: (x + 2)(3x + 5)\n",
            "First: Multiply the first term of each binomial:\n",
            "x*3x=3x"+superscript(2)+"\n",
            "Outer: Multiply the outermost terms in the expression:\n",
            "x*5=5x\n",
            "Inner: Multiply the innermost terms:\n",
            "2*3x=6x\n",
            "Last: Multiply the last term of each binomial:\n",
            "2*5=10\n",
            "\n",
            "Write out the sum of all four products:\n",
            "3x"+superscript(2)+"+5x+6x+10\n",
            "\n",
            "Combine the like terms:\n",
            "3x"+superscript(2)+"+11x+10\n",
            "\n",
            "If your polynomial has more than 2 factors (roots) FOIL the first two factors and then FOIL your new equation with the next factor\n",
            "and so on until every term has been multiplied by each other.\n",
             
            Button("Home", new_equation),
            Button("Back", skill_dashboard if state.path.manual_mode else graph_x_intercepts),
            Button("Next", expand_factored_page)
        ]
    )


@route
def expand_factored_page(state):
    return Page(state, [
        "Convert to Standard Form\n",
        "Equation:",
        state.factored_equation,
        "\n",
        "Enter the standard form (use ^ for exponents):",
        TextBox("answer"),
        Button("Submit", check_standard_form),
        "\n",
        Button("Home", new_equation),
        Button("Back", expand_factored_instructions)
    ])


@route
def check_standard_form(state, answer):
    answer = answer.replace(" ", "")
    answer = answer.replace("^2", "²").replace("^3", "³").replace("^4", "⁴")
    correct = build_standard_form(state.coefficients).replace(" ", "")
    if answer != correct:
        return Page(state, [
            "Incorrect. Try again.",
            Button("Try Again", expand_factored_page),
            Button("Back", expand_factored_instructions)
        ])
    state.standard_equation = correct
    state.path.expand_found = True
    return skill_dashboard(state)


@route
def degree_instructions(state):
    # Compare the end behavior of positive and negative leading coefficients.
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    x = np.linspace(-3, 3, 400)
    examples = [
        (x ** 2, "x²", "Even degree, Positive leading coefficient"),
        (x ** 3, "x³", "Odd degree, Positive leading coefficient"),
        (-x ** 2, "-x²", "Even degree, Negative leading coefficient"),
        (-x ** 3, "-x³", "Odd degree, Negative leading coefficient"),
    ]

    for ax, (y, title, description) in zip(axes.flat, examples):
        ax.plot(x, y)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(True)
        ax.set_title(title + "\n" + description)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()

    return Page(
        state,
        [
            "How to Find the Degree of a Polynomial\n",
            "\n",
            "The degree is the highest exponent of x in the polynomial when it is written in standard form.\n",
            "Example: f(x)=4x³-2x²+7x-5 has degree 3 because 3 is the highest exponent.\n",
            "\n",
            "The degree helps determine the graph's end behavior.\n",
            "Even-degree functions have the same end behavior on both sides: both ends rise or both ends fall. For example, y=x² has both ends rising.\n",
            "Odd-degree functions have different end behavior: one end rises and the other falls. For example, y=x³ falls to the left and rises to the right.\n",
            "\n",
            "The coefficient of the term with the highest degree is called the leading coefficient. Its sign determines the direction of the ends. A positive leading coefficient gives the usual upward-right behavior; a negative leading coefficient reverses the end behavior. Compare y=x² with y=-x² and y=x³ with y=-x³ in the graphs below.\n",
            "\n",
            "The degree also gives the maximum possible number of x-intercepts. A polynomial of degree n can have at most n real x-intercepts, but it may have fewer.\n",
            MatPlotLibPlot(),
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
    return Page(state, [
        "Equation:",
        state.standard_equation,
        "\n",
        "What is the degree of your polynomial?",
        "\n",
        TextBox("degree_answer"),
        "\n",
        "\n",
        "Is the degree Odd or Even?",
        "\n",
        TextBox("parity_answer"),
        Button("Submit", check_degree),
        "\n",
        Button("Home", new_equation),
        Button("Back", degree_instructions)
    ])


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

            f"Equation: {state.standard_equation}",

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
            f"Equation: {state.standard_equation}",
            MatPlotLibPlot(),
            Button("Home", new_equation)
        ]
    )

# =========================
# Start Program
# =========================

start_server(State())
