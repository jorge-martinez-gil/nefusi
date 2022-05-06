"""Main function"""
import logging
import argparse

from models.use_use import USECalculator
from models.use_elmo import ELMoCalculator
from models.use_bert import BERTCalculator


def main(config):
    models = {
        "use": USECalculator,
        "elmo": ELMoCalculator,
        "bert": BERTCalculator,
    }

    if config.model not in models:
        logging.error(f"The model you chosen is not supported yet.")
        return

    if config.verbose:
        logging.info(f"Loading the corpus...")

    sentences = []
    with open("geresid-source.txt", "r", encoding="utf-8") as corpus:
        
        #sentences = [sentence.replace("", "") for sentence in corpus.readlines()]
        

        for sentence in corpus.readlines():
            sentences.append (sentence)
        if config.verbose:
            logging.info(
                f'You chose the "{config.model.upper()}" as a model.\n'
                f'You chose the "{config.method.upper()}" as a method.'
            )

    par = []
    impar = []
    par = sentences[0::2]
    impar = sentences[1::2]
    sent2 = []
    
    # We print the column
    for i in range(50):
        sent2.append(par[i])
        sent2.append(impar[i])
        print (sent2)
        model = models[config.model](config, sent2)
        model.calculate()
        sent2.clear()
        
    # We read the column into a pandas
    
    

    if config.verbose:
       logging.info(f"Terminating the program...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="use", choices=["use", "elmo", "bert"]
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosine",
        choices=[
            "cosine",
            "manhattan",
            "euclidean",
            "inner",
            "ts-ss",
            "angular",
            "pairwise",
            "pairwise-idf",
        ],
    )
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()
    main(args)
