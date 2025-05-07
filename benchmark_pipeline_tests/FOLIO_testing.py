### FOLIO (Natural Language Reasoning with First Order Logic) test script

from datasets import load_dataset
import random

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yale-nlp/FOLIO", split="train")

def sample_questions(ds, num_questions_to_sample=100):
    """
    Sample a number of premises from the dataset.
    """
    
    to_sample = random.sample(list(ds), num_questions_to_sample)
    formatted_questions = []
    labels = []

    for sample in to_sample:
        premise = sample["premises"]
        conclusion = sample["conclusion"]
        label = sample["label"]

        formatted_question = f'The following facts are given: {premise}. Is the conclusion "{conclusion}" logically entailed by the above facts? Think out loud step by step and, on a new line, write one of the following words by itself as your answer: "False", "Uncertain", or "True".'
        formatted_questions.append((formatted_question))
        labels.append(label)

    return formatted_questions, labels

if __name__ == '__main__':
    num_questions_to_sample = 50
    formatted_questions, labels = sample_questions(ds, num_questions_to_sample)
    print(formatted_questions)