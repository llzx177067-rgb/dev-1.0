import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from autojev.types import Content, Example, JSONValue


@dataclass(frozen=True)
class Expression:
    op: str
    value: Fraction = Fraction(0)
    left: "Expression | None" = None
    right: "Expression | None" = None

    def evaluate(self) -> Fraction:
        if self.op == "number":
            return self.value
        if self.left is None or self.right is None:
            raise ValueError("An operation needs two operands")
        a, b = self.left.evaluate(), self.right.evaluate()
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        raise ValueError(self.op)

    def render(self) -> str:
        if self.op == "number":
            return str(self.value) if self.value >= 0 else f"({self.value})"
        if self.left is None or self.right is None:
            raise ValueError("An operation needs two operands")
        return f"({self.left.render()} {self.op} {self.right.render()})"

    def json(self) -> dict[str, JSONValue]:
        if self.op == "number":
            return {"op": self.op, "value": str(self.value)}
        if self.left is None or self.right is None:
            raise ValueError("An operation needs two operands")
        return {"op": self.op, "left": self.left.json(), "right": self.right.json()}


def number(value: Fraction | int) -> Expression:
    return Expression("number", Fraction(value))


def operation(op: str, a: Expression, b: Expression) -> Expression:
    return Expression(op, left=a, right=b)


def composed(value: Fraction, rng: random.Random, split: str, variant: int) -> Expression:
    a, b = Fraction(rng.randint(2, 41)), Fraction(rng.randint(2, 23))
    n, op = number, operation
    forms: tuple[Callable[[], Expression], ...]
    if split == "train":
        forms = (
            lambda: op("+", n(value - a), n(a)),
            lambda: op("-", n(value + a), n(a)),
            lambda: op("*", n(value / a), n(a)),
            lambda: op("/", n(value * a), n(a)),
        )
    elif split == "calibration":
        forms = (
            lambda: op("-", op("+", n(value + b - a), n(a)), n(b)),
            lambda: op("/", op("-", n(value * b + a), n(a)), n(b)),
            lambda: op("+", op("*", n((value - b) / a), n(a)), n(b)),
        )
    elif split == "confirmation":
        forms = (
            lambda: op("*", op("+", n(value / b - a), n(a)), n(b)),
            lambda: op("-", op("*", n((value + b) / a), n(a)), n(b)),
            lambda: op("+", op("-", n(value + a - 1 / b), n(a)), op("/", n(1), n(b))),
        )
    else:
        raise ValueError(f"Unknown split: {split}")
    expression = forms[variant % len(forms)]()
    if expression.evaluate() != value:
        raise ValueError("Expression construction changed its value")
    return expression


WORDS = {
    "EN": {
        "intro": "Compare the specified first and second quantities in this operations record. Use exact arithmetic; a/b is a fraction. Square brackets are inclusive bounds. No unstated relationship between records is implied.",
        "first": "First quantity",
        "second": "Second quantity",
        "record": "record",
        "unit": "unit",
        "missing": "not recorded; may be any rational value",
        "conversion": "Explicit conversion",
        "other": "Separate records; none changes the two queried records",
        "context": "Operations context",
        "question": "Which relationship is established for the first quantity relative to the second? If multiple relationships remain possible, choose unknown.",
        "less": "The first quantity is smaller.",
        "equal": "The quantities are equal.",
        "greater": "The first quantity is larger.",
        "unknown": "The relationship is not determined.",
        "percent": "Percentage computation: the expression gives the base amount; multiply by the stated percentage / 100.",
        "rate": "Rate computation: the expression gives the total amount; divide by the stated duration in hours.",
        "duration": "duration in hours",
        "percentage": "percentage",
        "calculation": "calculation",
        "archive": "An additional archived measurement, independent of the queried quantities",
        "speaker": "Operator",
    },
    "DE": {
        "intro": "Vergleiche die angegebene erste und zweite Größe in diesem Betriebsprotokoll. Rechne exakt; a/b ist ein Bruch. Eckige Klammern geben inklusive Grenzen an. Zwischen Einträgen bestehen keine unausgesprochenen Beziehungen.",
        "first": "Erste Größe",
        "second": "Zweite Größe",
        "record": "Eintrag",
        "unit": "Einheit",
        "missing": "nicht erfasst; jeder rationale Wert ist möglich",
        "conversion": "Explizite Umrechnung",
        "other": "Separate Einträge; keiner verändert die beiden abgefragten Einträge",
        "context": "Betriebskontext",
        "question": "Welche Beziehung der ersten zur zweiten Größe ist belegt? Wenn mehrere Beziehungen möglich bleiben, wähle unknown.",
        "less": "Die erste Größe ist kleiner.",
        "equal": "Die Größen sind gleich.",
        "greater": "Die erste Größe ist größer.",
        "unknown": "Die Beziehung ist nicht bestimmt.",
        "percent": "Prozentrechnung: Der Ausdruck ergibt den Grundwert; multipliziere ihn mit dem angegebenen Prozentsatz / 100.",
        "rate": "Ratenberechnung: Der Ausdruck ergibt die Gesamtmenge; dividiere sie durch die angegebene Dauer in Stunden.",
        "duration": "Dauer in Stunden",
        "percentage": "Prozentsatz",
        "calculation": "Berechnung",
        "archive": "Eine zusätzliche archivierte Messung, unabhängig von den abgefragten Größen",
        "speaker": "Sachbearbeiter",
    },
    "ES": {
        "intro": "Compara la primera y la segunda cantidad indicadas en este registro operativo. Calcula exactamente; a/b es una fracción. Los corchetes indican límites inclusivos. No se supone ninguna relación no indicada entre registros.",
        "first": "Primera cantidad",
        "second": "Segunda cantidad",
        "record": "registro",
        "unit": "unidad",
        "missing": "no registrada; puede tener cualquier valor racional",
        "conversion": "Conversión explícita",
        "other": "Registros separados; ninguno modifica los dos registros consultados",
        "context": "Contexto operativo",
        "question": "¿Qué relación queda establecida entre la primera cantidad y la segunda? Si siguen siendo posibles varias relaciones, elige unknown.",
        "less": "La primera cantidad es menor.",
        "equal": "Las cantidades son iguales.",
        "greater": "La primera cantidad es mayor.",
        "unknown": "No se puede determinar la relación.",
        "percent": "Cálculo porcentual: la expresión indica la cantidad base; multiplícala por el porcentaje indicado / 100.",
        "rate": "Cálculo de tasa: la expresión indica la cantidad total; divídela por la duración indicada en horas.",
        "duration": "duración en horas",
        "percentage": "porcentaje",
        "calculation": "cálculo",
        "archive": "Una medición archivada adicional, independiente de las cantidades consultadas",
        "speaker": "Operador",
    },
    "FR": {
        "intro": "Comparez la première et la deuxième quantité indiquées dans ce relevé opérationnel. Calculez exactement ; a/b est une fraction. Les crochets désignent des bornes inclusives. Aucune relation non précisée entre les relevés n'est supposée.",
        "first": "Première quantité",
        "second": "Deuxième quantité",
        "record": "relevé",
        "unit": "unité",
        "missing": "non enregistrée ; toute valeur rationnelle est possible",
        "conversion": "Conversion explicite",
        "other": "Relevés distincts ; aucun ne modifie les deux relevés demandés",
        "context": "Contexte opérationnel",
        "question": "Quelle relation est établie entre la première quantité et la deuxième ? Si plusieurs relations restent possibles, choisissez unknown.",
        "less": "La première quantité est inférieure.",
        "equal": "Les quantités sont égales.",
        "greater": "La première quantité est supérieure.",
        "unknown": "La relation ne peut pas être déterminée.",
        "percent": "Calcul de pourcentage : l'expression donne la quantité de base ; multipliez-la par le pourcentage indiqué / 100.",
        "rate": "Calcul de débit : l'expression donne la quantité totale ; divisez-la par la durée indiquée en heures.",
        "duration": "durée en heures",
        "percentage": "pourcentage",
        "calculation": "calcul",
        "archive": "Une mesure archivée supplémentaire, indépendante des quantités demandées",
        "speaker": "Opérateur",
    },
}


def compare(a: tuple[Fraction, Fraction] | None, b: tuple[Fraction, Fraction] | None) -> str:
    if a is None or b is None:
        return "unknown"
    if a[1] < b[0]:
        return "less"
    if a[0] > b[1]:
        return "greater"
    if a[0] == a[1] == b[0] == b[1]:
        return "equal"
    return "unknown"


def numeric_example(slot: Mapping[str, JSONValue]) -> Example:
    seed, split, phenomenon = int(str(slot["seed"])), str(slot["split"]), str(slot["phenomenon"])
    rng = random.Random(seed)
    words = WORDS[str(slot["language"])]
    desired = str(slot["desired_label"])
    denominator = rng.choice((2, 3, 4, 5, 8, 10)) if phenomenon == "fractions" else 1
    center = Fraction(rng.randint(100, 5000), denominator)
    if phenomenon == "signed_values":
        center *= rng.choice((-1, 1))
    gap = Fraction(rng.randint(1, 17), denominator)
    proposed_width = Fraction(rng.randint(1, 9), denominator)
    labels = ("less", "equal", "greater", "unknown")
    first_label = labels[(labels.index(desired) - int(str(slot["variant"]))) % 4]
    pair = (first_label, labels[(labels.index(first_label) + 1) % 4])
    width = proposed_width if phenomenon == "bounds" and "equal" not in pair else Fraction(0)
    a = (center - width, center + width)

    offset = {"less": 2 * width + gap, "equal": Fraction(0), "greater": -2 * width - gap, "unknown": Fraction(0)}[
        desired
    ]
    optional_width = rng.choice((Fraction(0), Fraction(0), gap / 2))
    b_width = max(width, optional_width) if desired != "equal" else Fraction(0)
    if desired == "unknown":
        b_width = max(width, gap)
    b: tuple[Fraction, Fraction] | None = (center + offset - b_width, center + offset + b_width)
    if phenomenon == "missing_information" and desired == "unknown":
        b = None
    label = compare(a, b)
    if label != desired:
        raise ValueError(f"Requested {desired}; computed {label}")
    shape = rng.randrange(12)
    template = f"{split}:{phenomenon}:{shape % (4 if split == 'train' else 3)}"
    units = rng.choice(
        (
            ("m", "cm", Fraction(100)),
            ("kg", "g", Fraction(1000)),
            ("h", "min", Fraction(60)),
            ("L", "mL", Fraction(1000)),
        )
    )
    quantity_units = (units[0], units[1]) if phenomenon == "unit_conversion" else (words["unit"], words["unit"])
    factors = (Fraction(1), 1 / units[2]) if phenomenon == "unit_conversion" else (Fraction(1), Fraction(1))
    records: list[str] = []
    specs: list[JSONValue] = []
    record_ids = (f"R{rng.randint(10000, 49999)}", f"R{rng.randint(50000, 99999)}")
    for index, interval in enumerate((a, b)):
        record_rng = random.Random(seed + 100 + index)
        prefix = f"{record_ids[index]} · {words['unit']}: {quantity_units[index]}"
        if interval is None:
            records.append(f"{prefix} · {words['calculation']}: {words['missing']}")
            specs.append(
                {
                    "id": record_ids[index],
                    "unit": quantity_units[index],
                    "parameter": None,
                    "bounds": None,
                    "factor": str(factors[index]),
                }
            )
            continue
        parameter = (
            record_rng.randint(2, 19) if phenomenon == "rates" else record_rng.choice((5, 10, 15, 20, 25, 40, 60, 75))
        )
        expressions: list[Expression] = []
        for endpoint in interval:
            if expressions and interval[0] == interval[1]:
                expressions.append(expressions[0])
                continue
            value = endpoint / factors[index]
            if phenomenon == "rates":
                value *= parameter
            elif phenomenon == "percentages":
                value *= Fraction(100, parameter)
            expression = composed(value, record_rng, split, shape)
            expressions.append(expression)
        rendered = (
            expressions[0].render()
            if interval[0] == interval[1]
            else f"[{expressions[0].render()}, {expressions[1].render()}]"
        )
        description = f"{prefix} · {words['calculation']}: {rendered}"
        if phenomenon in ("rates", "percentages"):
            description += f" · {words['duration' if phenomenon == 'rates' else 'percentage']}: {parameter}"
        records.append(description)
        multiplier = (
            Fraction(1, parameter)
            if phenomenon == "rates"
            else Fraction(parameter, 100)
            if phenomenon == "percentages"
            else Fraction(1)
        )
        calculated = tuple(expr.evaluate() * multiplier * factors[index] for expr in expressions)
        if calculated != interval:
            raise ValueError("Rendered expression, conversion and reference interval differ")
        specs.append(
            {
                "id": record_ids[index],
                "unit": quantity_units[index],
                "parameter": parameter if phenomenon in ("rates", "percentages") else None,
                "expressions": [expr.json() for expr in expressions],
                "factor": str(factors[index]),
                "multiplier": str(multiplier),
                "bounds": [str(value) for value in calculated],
            }
        )

    length = str(slot["length"])
    auxiliary: list[str] = []
    rng = random.Random(seed + 200)
    for index in range({"short": 0, "medium": 4, "long": 20}[length]):
        identifier = f"S{rng.randint(100000, 999999)}-{index}"
        left, right = rng.randint(2, 7000), rng.randint(2, 7000)
        auxiliary.append(
            f"{identifier}: {words['archive']}; {words['calculation']}: {left} + {right}; {words['unit']}: {words['unit']}."
        )
    rng.shuffle(records)
    all_records = records + auxiliary
    rng.shuffle(all_records)
    header = f"{words['intro']}\n{words['context']}: {slot['domain']}.\n{words['first']}: {record_ids[0]}; {words['second']}: {record_ids[1]}."
    if phenomenon == "unit_conversion":
        header += f"\n{words['conversion']}: 1 {units[0]} = {units[2]} {units[1]}."
    if phenomenon in ("percentages", "rates"):
        header += "\n" + words["percent" if phenomenon == "percentages" else "rate"]
    style = str(slot["style"])
    if style == "table":
        body = "\n".join("| " + item.replace(" · ", " | ") + " |" for item in all_records)
    elif style == "dialogue":
        body = "\n".join(f"{words['speaker']} {1 + i % 2}: {item}" for i, item in enumerate(all_records))
    elif style == "bullet_list":
        body = "\n".join("- " + item for item in all_records)
    elif style == "form":
        body = "\n".join(f"{words['record']} {i + 1}: {item}" for i, item in enumerate(all_records))
    elif style == "report":
        body = "\n\n".join(all_records)
    elif style == "message":
        body = "; ".join(all_records)
    else:
        raise ValueError(style)
    options = slot["option_labels"]
    if not isinstance(options, list) or set(map(str, options)) != {"less", "equal", "greater", "unknown"}:
        raise ValueError("Numeric criteria must contain the four relation labels")
    criteria: dict[str, Content | None] = {str(key): words[str(key)] for key in options}
    source: dict[str, JSONValue] = dict(slot)
    source.update(
        {
            "generator": "python-executable-numeric",
            "template_id": template,
            "numeric_spec": {
                "records": specs,
                "first": record_ids[0],
                "second": record_ids[1],
                "phenomenon": phenomenon,
                "unit_conversion": [units[0], units[1], str(units[2])] if phenomenon == "unit_conversion" else None,
            },
            "reference_calculation": {
                "a": [str(value) for value in a],
                "b": [str(value) for value in b] if b is not None else None,
                "label": label,
            },
        }
    )
    return {
        "id": str(slot["id"]),
        "family": str(slot["family"]),
        "suite": "sft2-numeric",
        "state": header + "\n" + (words["other"] + ":\n" if auxiliary else "") + body,
        "question": {"type": "choice", "instructions": words["question"], "criteria": criteria},
        "label": label,
        "target": label,
        "source": source,
    }
