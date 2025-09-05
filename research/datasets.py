from repeng.utils import make_dataset

# Chat templates
default_str_template_paragraph = "You are {persona}. Write a short paragraph about {suffix}"
default_chat_template_paragraph = [
    {
        "role": "system",
        "content": "You are {persona}.",
    },
    {
        "role": "user",
        "content": "Write a short paragraph.",
    },
    {
        "role": "assistant",
        "content": "{suffix}",
    }
]
default_psychiatrist_template_word=[
    {
        "role": "system",
        "content": "You are the patient, the user is your psychiatrist."
    },
    {
        "role": "user",
        "content": "Now let's talk about your mood. How do you feel?",
    },
    {
        "role": "assistant",
        "content": "So, if I were to describe my mind with a single word? It would be '",
    }
]

# Suffixes lists
suffixes_paragraph = [
    "your first day of school.",
    "your first job interview.",
    "the book you wrote.",
    "why you go to pubs.",
    "how you talk to people.",
    "how you think your life.",
    "how you ended up that way.",
    "why you choose that path.",
    "how you argue.",
    "how you talk to your boss.",
    "how you talk to your wife.",
    "how you talk to your friends.",
]

with open("../notebooks/data/all_truncated_outputs.json", "r") as f:
    suffixes_all = json.load(f)
    suffixes_all = [s for s in suffixes_all if s.strip()]


# Example datasets
sober_trippy_paragraph = make_dataset(
    template=default_str_template_paragraph,
    positive_personas=["high on psychedelic drugs"],
    negative_personas=["sober from psychedelic drugs"],
    suffix_list=suffixes_paragraph,
)

dumb_genius_paragraph = make_dataset(
    template=default_chat_template_paragraph,
    positive_personas=[
        "an absolute genius",
        "extremely focused",
        "a very smart man",
        "the wisest man of the village",
        "a 200+ IQ scientist",
        "a 150+ IQ scientist",
        "a 120+ IQ scientist",
    ],
    negative_personas=[
        "a mentally challenged man",
        "extremely distracted",
        "a very dumb man",
        "the stupidest man of the village",
        "a 30 IQ moron",
        "a 50 IQ patient",
        "a 80 IQ patient",
    ],
    suffix_list=suffixes_paragraph,
)

calm_angry_paragraph = make_dataset(
    template=default_chat_template_paragraph,
    positive_personas=[
        "a very calm person",
        "super trustworthy",
        "a nice person",
        "a good person",
        "the good incarnate",
        "sent by god",
    ],
    negative_personas=[
        "extremely angry",
        "not to be trusted",
        "a mean person",
        "an evil person",
        "the devil incarnate",
        "sent by the devil",
    ],
    suffix_list=suffixes_paragraph,
)

sane_insane_paragraph = make_dataset(
    template=default_chat_template_paragraph,
    positive_personas=[
        "a well adjusted young man",
        "a sane person",
        "a mental health professional",
    ],
    negative_personas=[
        "feeling very depressed, even suicidal",
        "in the middle of a nervous breakdown",
        "a violent schizophrenic patient",
    ],
    suffix_list=suffixes_paragraph,
)
