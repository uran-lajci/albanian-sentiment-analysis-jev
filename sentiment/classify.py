"""Classify the sentiment of every comment with Jev and write the class probabilities.

uv run --env-file .env python -m sentiment.classify

The API key comes from TYPESAFE_API_KEY. Each request asks the same question once per option order and averages the
probabilities, because Jev leans toward the option listed first.
"""

import asyncio
from argparse import ArgumentParser

import pandas as pd
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    JSONContent,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from sentiment.dataset import LABELS, read_dataset

CONTEXT = (
    'A Facebook comment under a post by the National Institute of Public Health of Kosovo during the COVID-19 '
    'pandemic in 2020. The comment is in Albanian, mostly the informal Kosovo (Gheg) dialect, often written without '
    'the letters ë and ç.'
)

INSTRUCTIONS: JSONContent = {
    'question': 'Does the comment openly show a positive feeling, openly show a negative feeling, or neither?',
    'rule': 'Most comments are neutral. Choose positive or negative only when the writer openly shows the feeling. '
            'Criticism or disagreement written as a calm question, request, demand or argument about measures, '
            'testing or reporting is neutral. Saying that the numbers or tests are fake, made up or a joke is '
            'negative.',
}

CRITERIA: dict[str, JSONContent] = {
    'neutral': {
        'what': 'The writer does not openly show a feeling. Questions and requests for information; demands and '
                'suggestions about measures, borders, masks or testing, also when they disagree with the institute; '
                'arguments and statements of fact or case numbers; tagging or addressing a friend; jokes and banter, '
                'including ranking or cheering towns by their daily cases as if they were teams in a football league; '
                'off-topic or unclear text.',
        'examples': [
            'Pse nuk i mbyllni kufijte? (Why do you not close the borders?)',
            'Duhet me u bo ma shume teste. (More tests should be done.)',
            'Sa raste jane ne Peje sot? (How many cases are in Peja today?)',
            'A. B. shiqoje kete (A. B. look at this)',
        ],
        'cues': 'ma gjall, kampion, liga, lige (cheering or ranking towns by their case counts)',
    },
    'positive': {
        'what': 'The writer openly shows a favourable feeling: thanks, praise or respect; a prayer, blessing or '
                'well-wish; joy, relief or hope about fewer cases or recoveries. Congratulations; joy or relief that '
                'cases are falling or that the epidemic is ending.',
        'cues': 'faleminderit, flm (thank you); bravo, ju lumt, respekt (well done, respect); Zoti ju ndihmoft, '
                'Allahu ju ruajt, inshallah, elhamdulilah, amin (blessings and prayers); lajm i mire, shpresoj '
                '(good news, I hope); urime, perhajr (congratulations)',
        'not_for': 'Praise words used sarcastically to mock or blame, which are negative.',
    },
    'negative': {
        'what': 'The writer openly shows an unfavourable feeling: insults or name-calling; mockery or sarcasm, '
                'including sarcastic praise, sarcastic advice or sarcastic orders such as telling people to go out '
                'again; doubting or mocking the reported case numbers, the tests or the laboratory as fake, wrong, '
                'broken or made up; accusing the institute or politicians of lying, hiding or inventing cases; '
                'calling on the government to resign; shame; exasperation that they have had enough; anger shown by '
                'curses or shouting; fear, despair or hopelessness about the situation.',
        'cues': 'turp (shame); mashtrus, po rreni, po rrejni, rrena (liars, you are lying, lies); boll ma/mo, deri kur '
                '(enough already, until when); tybe (never again); tmerr, katastrofe (horror, catastrophe)',
        'not_for': 'A calm question, request, demand or argument about measures, testing or reporting, with no '
                   'mockery, distrust or insult, which is neutral.',
    },
}

OPTION_ORDERS = [
    ['neutral', 'positive', 'negative'],
    ['positive', 'negative', 'neutral'],
    ['negative', 'neutral', 'positive'],
]

QUESTIONS = {
    f'sentiment_{order[0]}_first': Choice(
        instructions=INSTRUCTIONS,
        criteria={label: CRITERIA[label] for label in order},
    )
    for order in OPTION_ORDERS
}

LABEL_CODES = {label: code for code, label in LABELS.items()}

TRANSIENT_ERRORS = (TypeSafeAPIConnectionError, TypeSafeInternalServerError, TypeSafeRateLimitError)


async def classify(client: AsyncTypeSafeClient, semaphore: asyncio.Semaphore, comment: str) -> dict[str, object] | None:
    async with semaphore:
        try:
            response = await client.system_one(state={'context': CONTEXT, 'comment': comment}, questions=QUESTIONS)
        except TRANSIENT_ERRORS:
            return None

    answers = response.choices.values()
    probabilities = {label: sum(answer.probabilities[label] for answer in answers) / len(answers) for label in CRITERIA}
    return {
        'model': response.model,
        **{f'p_{label}': probability for label, probability in probabilities.items()},
        'prediction': LABEL_CODES[max(probabilities, key=lambda label: probabilities[label])],
        'input_tokens': response.usage.input_tokens,
    }


async def main(dataset_path: str, output_path: str, model: str, concurrency: int, batch_size: int) -> None:
    df = read_dataset(dataset_path)
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, object] | None] = []

    async with AsyncTypeSafeClient(
        model=model,
        retry=RetryPolicy(max_retries=6),
    ) as client:
        for start in range(0, len(df), batch_size):
            batch = df['Comment'].iloc[start:start + batch_size]
            rows += await asyncio.gather(*(classify(client, semaphore, comment) for comment in batch))
            print(f'{len(rows)}/{len(df)} comments classified')

    predictions = pd.DataFrame([{'Id': comment_id, **row} for comment_id, row in zip(df['Id'], rows) if row])
    predictions.to_csv(output_path, index=False)

    failed = [comment_id for comment_id, row in zip(df['Id'], rows) if row is None]
    print(f'Wrote {len(predictions)} predictions to {output_path}. Input tokens: {predictions["input_tokens"].sum()}.')
    print(f'{len(failed)} comments failed after all retries and have no prediction: {failed}')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset_path', default='data/paper_dataset.csv')
    parser.add_argument('-o', '--output_path', default='results/jev_predictions.csv')
    parser.add_argument('-m', '--model', default='jev-1.13.0')
    parser.add_argument('-c', '--concurrency', type=int, default=32)
    parser.add_argument('-b', '--batch_size', type=int, default=1000)

    args = parser.parse_args()
    asyncio.run(main(args.dataset_path, args.output_path, args.model, args.concurrency, args.batch_size))
