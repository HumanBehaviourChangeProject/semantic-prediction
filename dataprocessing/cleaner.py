import csv
from thefuzz import process as fw_process, fuzz
import json
from abc import ABC, abstractmethod

_MAPPINGS = set()


class Dropper:
    def __init__(self):
        with open("data/deleted.txt", "r") as fin:
            self.deleted_documents = [int(d.strip()) for d in fin]

    def should_be_dropped(self, doc_id):
        return doc_id in self.deleted_documents


class Differ:
    def __init__(self):
        with open("data/diff.json", "r") as fin:
            self.diff = json.load(fin)

    def get_diff(self, doc_id, arm_id):
        return self.diff[str(doc_id)][str(arm_id)]


def load_countries_and_cities():
    with open("data/worldcities.csv", "r") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        city_dict = {line[1]: line[4] for line in reader}

    return set(city_dict.values()).union({"USA", "UK", "England"}), city_dict


class AttributeCleaner(ABC):
    def __init_subclass__(cls, **kwargs):
        _MAPPINGS.add(cls())

    @property
    def linked_attributes(self):
        raise NotImplementedError

    def __call__(self, ident, data, diff):
        return self.apply_diff(self.get_value(ident, data), diff)

    def apply_diff(self, values, diff):
        return values

    @abstractmethod
    def get_value(self, ident, data):
        raise NotImplementedError

class RoundingCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080481, 6080485, 6080486, 6080512, 6080719)

    def get_value(self, ident, data):
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
                elif len(s) > 1:
                    print(f"Multiple values found in {ident}:", s)
        if v:
            return {ident: v}
        else:
            return {}


class PresenceCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        # Features that are commented out, are not present in the dataset
        return (
            6451788,
            6823485,
            6823485, # CBT
            #(6452744, 6452744)
            6452745, # 1.1.Goal setting (behavior)
            6452746, # 1.2 Problem solving
            6452747, # 1.3 Goal setting (outcome)
            6452748, # 1.4 Action planning
            6452749, # 1.5 Review behavior goal(s)
            6452750, # 1.6 Discrepancy between current behavior and goal
            6452751, # 1.7 Review outcome goal(s)
            6452752, # 1.8 Behavioral contract
            6452753, # 1.9 Commitment
            6452755, # 2.1 Monitoring of behavior by others without feedback
            6452756, # 2.2 Feedback on behaviour
            6452757, # 2.3 Self-monitoring of behavior
            6452758, # 2.4 Self-monitoring of outcome(s) of behaviour
            6452759, # 2.5 Monitoring of outcome(s) of behavior without feedback
            6452760, # 2.6 Biofeedback
            6452761, # 2.7 Feedback on outcome(s) of behavior
            #(6452762, 6452762)
            6452763, # 3.1 Social support (unspecified)
            6452764, # 3.2 Social support (practical)
            6452765, # 3.3 Social support (emotional)
            #(6452830, 6452830)
            6452831, # 4.1 Instruction on how to perform the behavior
            6452832, # 4.2 Information about Antecedents
            6452833, # 4.3 Re-attribution
            #(6452834, 6452834)
            6452836, # 4.5. Advise to change behavior
            #(6452837, 6452837)
            6452838, # 5.1 Information about health consequences
            6452839, # 5.2 Salience of consequences
            6452840, # 5.3 Information about social and environmental consequences
            6452843, # 5.4 Monitoring of emotional consequences
            #(6452844, 6452844)
            #(6452845, 6452845)
            #(6452846, 6452846)
            6452847, # 6.1 Demonstration of behavior
            6452848, # 6.2 Social comparison
            6452849, # 6.3 Information about others' approval
            #(6452850, 6452850)
            6452851, # 7.1 Prompts/cues
            #(6452852, 6452852)
            6452853, # 7.3 Reduce prompts/cues
            6452854, # 7.4 Remove access to the reward
            6452855, # 7.5 Remove aversive stimulus
            6452856, # 7.6 Satiation
            6452857, # 7.7 Exposure
            6452858, # 7.8 Associative learning
            #(6452859, 6452859)
            6452860, # 8.1 Behavioral practice/rehearsal
            6452861, # 8.2 Behavior substitution
            #(6452862, 6452862)
            6452863, # 8.4 Habit reversal
            #(6452864, 6452864)
            #(6452865, 6452865)
            6452930, # 8.7 Graded tasks
            #(6452931, 6452931)
            6452932, # 9.1 Credible source
            6452933, # 9.2 Pros and cons
            6452934, # 9.3 Comparative imagining of future outcomes
            #(6452935, 6452935)
            6452936, # 10.1 Material incentive (behavior)
            6452937, # 10.2 Material reward (behavior)
            6452938, # 10.3 Non-specific reward
            6452939, # 10.4 Social reward
            6452940, # 10.5 Social incentive
            6452941, # 10.6 Non-specific incentive
            #(6452942, 6452942)
            6452943, # 10.8 Incentive (outcome)
            6452944, # 10.9 Self-reward
            6452945, # 10.10 Reward (outcome)
            6452946, # 10.11 Future punishment
            #(6452947, 6452947)
            6452948, # 11.1 Pharmacological support
            6452949, # 11.2 Reduce negative emotions
            #(6452950, 6452950)
            #(6452952, 6452952)
            #(6452953, 6452953)
            6452954, # 12.1 Restructuring the physical environment
            6452955, # 12.2 Restructuring the social environment
            6452956, # 12.3 Avoidance/reducing exposure to cues for the behavior
            6452957, # 12.4 Distraction
            6452959, # 12.5 Adding objects to the environment
            6452960, # 12.6 Body changes
            #(6452961, 6452961)
            6452962, # 13.1 Identification of self as role model
            6452963, # 13.2 Framing/reframing
            6452964, # 13.3 Incompatible beliefs
            6452965, # 13.4 Valued self-identify
            6452966, # 13.5 Identity associated with changed behavior
            #(6452967, 6452967)
            6452968, # 14.1 Behavior cost
            6452969, # 14.2 Punishment
            6452970, # 14.3 Remove reward
            6452973, # 14.4 Reward approximation
            #(6452974, 6452974)
            #(6452975, 6452975)
            #(6452976, 6452976)
            6452977, # 14.8 Reward alternative behavior
            6452978, # 14.9 Reduce reward frequency
            #(6452979, 6452979)
            #(6452980, 6452980)
            6452981, # 15.1 Verbal persuasion about capability
            6452982, # 15.2 Mental rehearsal of successful performance
            6452983, # 15.3 Focus on past success
            6452984, # 15.4 Self-talk
            #(6452985, 6452985)
            6452986, # 16.1 Imaginary punishment
            6452987, # 16.2 Imaginary reward
            6080701,
            6080686,
            6080692,
            6080694,
            6080688,
            6830264,
        )

    def get_value(self, ident, data):
        x = data.get(ident, False)
        if x:
            v = 1
        else:
            v = 0
        return {ident: v}


class OutcomeValueCleaner(RoundingCleaner):

    @property
    def linked_attributes(self):
        return (6451791,)

    def apply_diff(self, values, diff):
        new_ov = diff.get("NEW Outcome value")
        v = values.get(6451791)
        if new_ov:
            v = new_ov[1]
        if v is not None:
            values[6451791] = float(v.replace(",","."))
        return values

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


class MotivationalIntervewingCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6823487,)

    def get_value(self, ident, data):
        v = 0
        brief = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x.lower(), ("brief advice", "ba"))
            if match[0][1] >= 80:
                brief = 1
            else:
                v = 1

        return {ident: v, "brief advise": brief}


class DigitalContentCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080691,)

    def get_value(self, ident, data):
        v = 0
        text_message = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x.lower(), ("text", "text message"))
            if match[0][1] >= 80:
                text_message = 1
            else:
                v = 1

        return {ident: v, "text messaging": text_message}


class DistanceCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080687,)

    def get_value(self, ident, data):
        keys = ("phone", "call", "telephone", "quitline", "hotline")
        phone = 0
        for x, _ in data.get(ident, tuple()):
            match = fw_process.extract(x, keys)
            if match[0][1] >= 90:
                phone = 1
        return {"phone": phone}


class SomaticCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080693,)

    def get_value(self, ident, data):
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
        d.update(self.any_as_presence.get_value(ident, data))
        d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
        patch = self.any_as_presence.get_value(6080694, data)[6080694]
        if d["gum"] or d["lozenge"] or d["e_cigarette"] or patch or d["inhaler"]:
            d["nrt"] = 1
        return d


class PillCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080695,)

    def get_value(self, ident, data):
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
            "varenicline": (
                ["varenicline", "varen", "chantix", "champix"],
                True,
                False,
            ),
        }
        d = self.any_as_presence.get_value(ident, data)
        for x, _ in data.get(ident, tuple()):
            for key, (patterns, ands, ors) in keys.items():
                match = fw_process.extract(x, patterns)
                if (match[0][1] > 90 or ors) and ands:
                    d[key] = 1
        return d


class HealthProfessionalCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6080704,)

    def get_value(self, ident, data):
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
        d = self.any_as_presence.get_value(ident, data)
        d = _process_with_key_list(ident, keys, data, initial_dictionary=d)
        return d


class PsychologistCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080706,)

    def get_value(self, ident, data):
        keys = {"pychologist": ["pychologist", "psychol"]}
        d = _process_with_key_list(ident, keys, data)
        return d


class PatientRoleCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080508,)

    def get_value(self, ident, data):
        keys = {"aggregate patient role": ["patient"]}
        d = {"aggregate patient role": 0}
        d = _process_with_key_list(
            ident, keys, data, initial_dictionary=d, threshold=80
        )
        return d


class HealthCareFacilityCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return (6080629,)

    def get_value(self, ident, data):
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


class PharmacologicalInterestCleaner(AttributeCleaner):
    def __init__(self):
        self.any_as_presence = PresenceCleaner()

    @property
    def linked_attributes(self):
        return (6830268,)

    def get_value(self, ident, data):
        return {
            ident: self.any_as_presence.get_value(ident, data).get(ident, 0)
            or self.any_as_presence.get_value(6830264, data).get(6830264, 0)
        }


class TimePointCleaner(AttributeCleaner):
    def __init__(self):
        self.use_rounded = RoundingCleaner()

    @property
    def linked_attributes(self):
        return (6451782,)

    def apply_diff(self, values, diff):
        v = values.get("Combined follow up")
        tpv = diff.get("Manually added follow-up duration value")
        tpu = diff.get("Manually added follow-up duration units")
        if tpu and tpv:
            if tpu[1] == "days":
                factor = 1/7
            elif tpu[1] == "weeks":
                factor = 1
            elif tpu[1] == "months":
                factor = 4.35
            elif tpu[1] in ("years", "year"):
                factor = 4.35*12
            else:
                raise Exception(f"Unknown unit {tpu[1]}")
            v =  float(tpv[1].replace(",",".")) * factor
        if v is not None:
            values["Combined follow up"] = float(v)
        return values

    def get_value(self, ident, data):
        v = data.get(6451782) or data.get(6451773)
        k = "Combined follow up"
        if v:
            return self.use_rounded.get_value(k, {k: v})
        else:
            return {k: v}


class CountryCleaner(AttributeCleaner):
    def __init__(self):
        self.countries, self.city_dict = load_countries_and_cities()

    @property
    def linked_attributes(self):
        return (6080518,)

    def get_value(self, ident, data):
        countries_values = data.get(ident, set())
        if countries_values:
            countries_values = [x for x, _ in countries_values]
            if len(countries_values) > 1:
                value = "multinational"
            else:
                match, quality = fw_process.extract(
                    countries_values[0], self.countries, scorer=fuzz.ratio
                )[0]
                if quality > 80:
                    value = match
                else:
                    value = countries_values[0]

        else:
            regions = data.get(6080519, set())
            value = [self.city_dict[x] for x, _ in regions if x in self.city_dict]
            if value:
                value = value[0]
            else:
                return dict()
        return {ident: value}


class PregnancyTrialCleaner(AttributeCleaner):

    @property
    def linked_attributes(self):
        return tuple()

    def get_value(self, ident, data):
        d = {
            "Pregnancy trial": 0,
            "Pregnancy trial (Mixed)": 0,
        }
        v = data.get("Pregnancy trial 1 = yes, 2 = mix pg and non-pg")
        if v is not None:
            if v == "1":
                d["Pregnancy trial"] = 1
            elif v == "2":
                d["Pregnancy trial (Mixed)"] = 1
            else:
                raise ValueError("Unexpected value", v)

        return d


class RelapsePreventionTrialCleaner(AttributeCleaner):
    @property
    def linked_attributes(self):
        return tuple()

    def get_value(self, ident, data):
        d = {
            "Relapse Prevention Trial": 0,
            "Relapse Prevention Trial(Mixed)": 0,
        }
        v = data.get("Relapse prevention trial (1 = yes, 2 = mix of quitters and non-abstinent)")
        if v is not None:
            if v == "1":
                d["Relapse Prevention Trial"] = 1
            elif v == "2":
                d["Relapse Prevention Trial(Mixed)"] = 1
            else:
                raise ValueError("Unexpected value", v)

        return d

def get_id(s):
    return int(s.split("___")[1])


def clean_row(row, diff):
    values = {
        "bupropion": 0,
        "varenicline": 0,
        "pychologist": 0,
        "doctor": 0,
        "nurse": 0,
    }

    for cleaner in _MAPPINGS:
        for attribute_id in cleaner.linked_attributes:
            mapped = cleaner(attribute_id, row, diff)
            mapped_with_context = {k: mapped.get(k, 0) for k in mapped}
            values.update(mapped_with_context)
    return values
