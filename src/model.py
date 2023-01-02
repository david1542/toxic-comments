from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import labels


def load_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
