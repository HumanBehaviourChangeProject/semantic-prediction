def use_rounded(ident, data):
    v = None
    if ident in data:
        for i in reversed(range(0, 3)):
            s = set(
                round(float(x), i)
                for x, _ in data[ident]
                if _clean(x) not in ("no", "none") and is_number(x)
            )
            if len(s) == 1:
                return {ident: s.pop()}
    if v:
        return {ident: v}
    else:
        return dict()


def any_as_presence(ident, data):
    x = data.get(ident, False)
    if x:
        v = 1
    else:
        v = 0
    return {ident: v}


def _clean(x):
    return x.replace(",", "").replace(";", "").replace("-", "").replace(" ", "").lower()


def _process_with_key_list(
    ident, keys, data, initial_dictionary=None, threshold=90, negative=False
):
    if initial_dictionary:
        d = dict(initial_dictionary)
    else:
        d = dict()
    for x, _ in data.get(ident, tuple()):
        for key, patterns in keys.items():
            match = fw_process.extract(x, patterns)
            if match[0][1] >= threshold:
                d[key] = 1 if not negative else 0
    return d


def process_motivational_interviewing(ident, data):
    v = 0
    brief = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x.lower(), ("brief advice", "ba"))
        if match[0][1] >= 80:
            brief = 1
        else:
            v = 1

    return {ident: v, "brief advise": brief}


def process_digital_content(ident, data):
    v = 0
    text_message = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x.lower(), ("text", "text message"))
        if match[0][1] >= 80:
            text_message = 1
        else:
            v = 1

    return {ident: v, "text messaging": text_message}


def process_distance(ident, data):
    keys = ("phone", "call", "telephone", "quitline", "hotline")
    phone = 0
    for x, _ in data.get(ident, tuple()):
        match = fw_process.extract(x, keys)
        if match[0][1] >= 90:
            phone = 1
    return {"phone": phone}


def process_somatic(ident, data):
    keys = {
        "gum": ["gum", "polacrilex"],
        "lozenge": ["lozenge"],
        "e_cigarette": ["ecig", "ecigarette"],
        "inhaler": ["inhaler", "inhal"],
        "placebo": ["placebo"],
        "varenicline": ["varenicline", "varen", "chantix", "champix"],
        "nasal_spray": ["nasal"],
        "rimonabant": ["rimonab"],
        "nrt": ["nicotine", "nrt"],
        "bupropion": ["bupropion"],
    }
    d = dict(
        gum=0, lozenge=0, e_cigarette=0, inhaler=0, placebo=0, nasal_spray=0, nrt=0
    )
    d.update(any_as_presence(ident, data))
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
    patch = any_as_presence(6080694, data)[6080694]
    if d["gum"] or d["lozenge"] or d["e_cigarette"] or patch or d["inhaler"]:
        d["nrt"] = 1
    return d


def process_pill(ident, data):
    keys = {
        "bupropion": (
            ["bupropion"],
            True,
            any(
                True
                for x, _ in data.get(6080693, tuple())
                if _clean(x) in ["bupropion"]
            ),
        ),
        "nortriptyline": (["nortript"], True, False),
        "varenicline": (["varenicline", "varen", "chantix", "champix"], True, False),
    }
    d = any_as_presence(ident, data)
    for x, _ in data.get(ident, tuple()):
        for key, (patterns, ands, ors) in keys.items():
            match = fw_process.extract(x, patterns)
            if (match[0][1] > 90 or ors) and ands:
                d[key] = 1
    return d


def process_health_professional(ident, data):
    keys = {
        "nurse": ["nurse"],
        "doctor": [
            "physician",
            "doctor",
            "physician",
            "cardiologist",
            "pediatrician",
            "general pract",
            "GP",
            "resident",
            "internal medicine",
        ],
    }
    d = any_as_presence(ident, data)
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
    return d


def process_pychologist(ident, data):
    keys = {"pychologist": ["pychologist", "psychol"]}
    d = _process_with_key_list(ident, keys, data)
    return d


def process_aggregate_patient_role(ident, data):
    keys = {"aggregate patient role": ["patient"]}
    d = {"aggregate patient role": 0}
    d = _process_with_key_list(ident, keys, data, initial_dictionary=d, threshold=80)
    return d


def process_healthcare_facility(ident, data):
    keys = {"healthcare facility": ["smok"]}
    d = _process_with_key_list(
        ident,
        keys,
        data,
        initial_dictionary={
            "healthcare facility": 1 if data.get(ident, tuple()) else 0
        },
        negative=True,
    )
    return d


def is_number(x):
    if x is None:
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False


def process_proportion_female(ident, data):
    value = None
    females = use_rounded(
        ident, {ident: {x for x in data.get(ident, set()) if is_number(x[0])}}
    ).get(ident, None)
    if females:
        value = females
    else:
        males = {float(x[0]) for x in data.get(6080486, set()) if is_number(x[0])}
        if males:
            value = 100 - float(males.pop())
    if value:
        return {ident: value}
    else:
        return dict()


def process_pharmacological_interest(ident, data):
    return {
        ident: any_as_presence(ident, data).get(ident, 0)
        or any_as_presence(6830264, data).get(6830264, 0)
    }


def build_process_country():
    countries, city_dict = load_countries_and_cities()

    def inner(ident, data):
        countries_values = data.get(ident, set())
        if countries_values:
            countries_values = [x for x, _ in countries_values]
            if len(countries_values) > 1:
                value = "multinational"
            else:
                match, quality = fw_process.extract(
                    countries_values[0], countries, scorer=fuzz.ratio
                )[0]
                if quality > 80:
                    value = match
                else:
                    value = countries_values[0]

        else:
            regions = data.get(6080519, set())
            value = [city_dict[x] for x, _ in regions if x in city_dict]
            if value:
                value = value[0]
            else:
                return dict()
        return {ident: value}

    return inner


mappings = {
    6451791: use_rounded,
    0: use_rounded,
    6451788: any_as_presence,
    6823485: any_as_presence,
    6823487: process_motivational_interviewing,
    6452745: any_as_presence,
    6452746: any_as_presence,
    6452748: any_as_presence,
    6452756: any_as_presence,
    6452757: any_as_presence,
    6452838: any_as_presence,
    6452840: any_as_presence,
    6452948: any_as_presence,
    6452949: any_as_presence,
    6452763: any_as_presence,
    6452831: any_as_presence,
    6452836: any_as_presence,
    6080701: any_as_presence,
    6080686: any_as_presence,
    6080687: process_distance,
    6080692: any_as_presence,
    6080694: any_as_presence,
    6080693: process_somatic,
    6080688: any_as_presence,
    6080691: process_digital_content,
    6080695: process_pill,
    6080704: process_health_professional,
    6080706: process_pychologist,
    6080508: process_aggregate_patient_role,
    6080481: use_rounded,
    6080485: process_proportion_female,
    6080512: use_rounded,
    6080518: build_process_country(),  # Country of intervention
    6080629: process_healthcare_facility,
    6830264: any_as_presence,
    6830268: process_pharmacological_interest,
    6080719: use_rounded,
}
